# User-owned visual verification

The user performs visual and browser testing. All agents, including delegated agents, must not use Computer Use/CUA, browser-control tools, Playwright, Selenium, Puppeteer, headless browsers or equivalent UI automation for inspection, navigation, screenshots or testing. Do not open browser tabs automatically or duplicate the user's visual review. Do not bypass this restriction through a shell, another tool, worker or renderer. Verify code with appropriate static, unit, domain, server and HTTP/API checks, and handle user findings through the project's testing inbox or chat. State unverified UI behavior honestly. This applies unless the user explicitly changes this instruction.

## Project coordination

The accepted project memory is in `.vibemachina/`. MAIN alone edits it and performs Git writes. Each implementation worker owns explicitly assigned paths. New workers use GPT-5.6 Sol as requested by the user on 2026-10-17; already running workers continue unchanged. Public product name is Latin `Lado`, centralized in `contracts/product.json`; the `navidu` Python namespace is retained internally for compatibility with the prepared plan.

Bank videos, extracted frames and video descriptions must not be submitted to external AI services. Inference and data processing run locally or on the explicitly authorized self-hosted Brev instance `fierce-aquamarine-beaver`; do not commit data or model weights. All further training and heavy model runs belong on that server per the user's instruction. No slides, pitch scripts, presentation kit or promotional video in this assignment.
