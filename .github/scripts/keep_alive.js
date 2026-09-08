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
    // Let the app fully finish booting inside the iframe.
    await page.waitForTimeout(30000);
    console.log('Wake sequence complete.');
  } else {
    console.log('App was already awake.');
    await page.waitForTimeout(5000);
  }

  // Check if an iframe is present — a good-enough signal that the app loaded.
  // We skip innerHTML size checks because Streamlit iframes are cross-origin
  // and frame.evaluate() always throws a security error in that context.
  const iframeEl = await page.$('iframe');
  if (iframeEl) {
    console.log('iframe found — app appears awake and loaded.');
  } else {
    // No iframe yet is expected right after a wake click; the 30s wait above
    // usually covers it, but cold boots can take longer. Log a warning only —
    // the app was poked and will continue booting on its own.
    console.log('WARNING: no iframe found on final check — app may still be warming up.');
  }

  await browser.close();
})();
