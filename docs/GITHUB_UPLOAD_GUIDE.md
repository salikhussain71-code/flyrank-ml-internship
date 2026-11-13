# GitHub upload and Pages guide (website only, no commands)

## Extract (Windows)
Right-click `flyrank-ml-internship.zip` > Extract All. Open the extracted `flyrank-ml-internship` folder.

## Create the repository
1. Sign in at github.com as `salikhussain71-code`.
2. Click **+** > **New repository**. Name: `flyrank-ml-internship`. Choose **Public**. Do not add a README. Click **Create repository**.

## Upload
1. On the empty repo page click **uploading an existing file**.
2. Open the extracted folder, press Ctrl+A inside it, and drag everything into the browser. (Drag the *contents*, so `README.md` is at the repo root.)
3. Hidden folders (`.github`) may not upload by drag-and-drop. If missing, use **Add file > Create new file** and paste `.github/workflows/tests.yml`, or skip it (optional).
4. Confirm `.env` is NOT in the list. Click **Commit changes**.

## Enable GitHub Pages
1. Repo > **Settings** > **Pages**.
2. Source: **Deploy from a branch**. Branch: `main`, folder: `/ (root)`. Save.
3. Wait 1 to 3 minutes. The paper should appear at:
   `https://salikhussain71-code.github.io/flyrank-ml-internship/paper/`
4. Open it in a private window and confirm it loads. Only then is it PAPER-DEPLOYED.

## Record the URL
Edit `submission/paper_url.txt` on GitHub so it contains exactly one line: the verified URL.

## Submit to FlyRank
Capstone page > Add submission > Deliverable links: paste your repo URL `https://github.com/salikhussain71-code/flyrank-ml-internship`. FlyRank finds the paper through `submission/paper_url.txt`. Submit only after real results are in the paper (not DRAFT).
