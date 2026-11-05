# 012: Public site without asking IMD; email alerts to the owner

**Status:** decided by the owner, 5 November 2026.

## Decision

- **The forecast site is public on GitHub Pages**, including during shadow mode.
- **IMD is not being asked for permission.** I recommended asking and holding the launch until it replied; the owner decided to go ahead.
- **Alerts are email only, to the owner only.** The account and recipient are GitHub secrets, because the repo is public. No Telegram.

## What the check found (5 November 2026, a reading of public pages, not legal advice)

- The gridded temperature page asks for a citation (Srivastava et al. 2009) and disclaims liability. It has no licence or redistribution terms.
- IMD's Certificate of Undertaking bars passing data on or putting it on the internet without written approval. The data portal requires it from users procuring data there. We did not procure through the portal or sign it, and the only copy read was a 2018 one from IMD Lucknow.
- The free data series does not include gridded data, so there is no open-data licence to rely on.
- No rule was found on forecasts derived from IMD data.

So nothing found forbids publishing these forecasts, and nothing clearly permits it. This risk is accepted, not resolved.

## Safeguards

- The site publishes forecast probabilities and tiers only: no IMD values and no raw weather data. A test checks that downloads have no IMD columns.
- Raw IMD files stay in the private dataset.
- IMD is credited with the citation, and the site says IMD has not reviewed or endorsed the forecast.
- Every page says it is not an official forecast or warning and points to IMD. The alert email says the same. This matters because IMD issues India's official heatwave warnings, and tier names such as "Warning" could be mistaken for theirs.

## Known and left as is

Public repo reports (for example `reports/label_qa.md`, `reports/label_drift_check.md`) already contain summaries computed from IMD data, such as heatwave shares and POWER-against-IMD comparisons. These are not the raw data.

If IMD objects, the site can be taken down the same day by disabling GitHub Pages, and those reports can be removed.
