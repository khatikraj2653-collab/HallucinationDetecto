const { chromium } = require('playwright');

(async () => {
  const url = process.env.APP_URL;
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
  } catch (e) {
    console.log('Initial networkidle wait timed out, continuing:', e.message);
  }

  // Streamlit's sleep interstitial is served at the top level (not inside
  // the app iframe), so check the outer page for the wake button.
  const wakeButton = page.getByRole('button', { name: /get this app back up/i });
  const isAsleep = await wakeButton.count().then((n) => n > 0).catch(() => false);

  if (isAsleep) {
    console.log('App is asleep — clicking wake button...');
    await wakeButton.first().click();
    // Waking takes a while: dependency install + app boot on Streamlit Cloud's side.
    try {
      await page.waitForSelector('iframe', { timeout: 90000 });
    } catch (e) {
      console.log('Timed out waiting for app iframe after wake click:', e.message);
    }
    // Let the app fully finish booting inside the iframe. A cold boot after
    // real hibernation (not just idle) can take a while.
    await page.waitForTimeout(30000);
  } else {
    console.log('App was already awake.');
    await page.waitForTimeout(5000);
  }

  // Final check: confirm the iframe now has real content, not still the sleep
  // screen. Streamlit's content isn't always exposed via innerText (it can
  // render into elements that don't register as visible text), so check
  // innerHTML size instead — a real loaded app is tens of KB; an empty or
  // still-booting shell is much smaller.
  const iframeEl = await page.$('iframe');
  if (iframeEl) {
    const frame = await iframeEl.contentFrame();
    const htmlLen = frame
      ? await frame.evaluate(() => document.body.innerHTML.length).catch(() => 0)
      : 0;
    console.log('iframe body innerHTML length:', htmlLen);
    if (htmlLen < 2000) {
      console.log('WARNING: iframe content looks too small — app may still be booting.');
      process.exitCode = 1;
    } else {
      console.log('App confirmed awake and loaded.');
    }
  } else {
    console.log('WARNING: no iframe found on final check — app may still be waking up.');
    process.exitCode = 1;
  }

  await browser.close();
})();
