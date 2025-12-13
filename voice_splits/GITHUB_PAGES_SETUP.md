# Hosting Survey on GitHub Pages

## Quick Setup (5 minutes)

### Step 1: Commit and Push Your Files

```bash
# Make sure you're in the project root
cd /path/to/Japanese_FastSpeech2

# Add the survey files
git add voice_splits/audio_survey_email.html
git add voice_splits/
git commit -m "Add TTS audio quality survey"

# Push to GitHub
git push origin master
```

### Step 2: Enable GitHub Pages

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/Japanese_FastSpeech2`
2. Click **Settings** (top right)
3. Scroll down to **Pages** (left sidebar)
4. Under "Source":
   - Select branch: **master** (or **main**)
   - Select folder: **/ (root)**
5. Click **Save**
6. Wait 1-2 minutes for deployment

### Step 3: Access Your Survey

Your survey will be available at:
```
https://YOUR_USERNAME.github.io/Japanese_FastSpeech2/voice_splits/audio_survey_email.html
```

**Example:**
If your username is `derekvawdrey`:
```
https://derekvawdrey.github.io/Japanese_FastSpeech2/voice_splits/audio_survey_email.html
```

---

## 📋 Create a Shorter Link (Optional)

### Option A: Create an index.html redirect

Create a file at the root: `index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=voice_splits/audio_survey_email.html">
    <title>Redirecting to Survey...</title>
</head>
<body>
    <p>Redirecting to survey... <a href="voice_splits/audio_survey_email.html">Click here if not redirected</a></p>
</body>
</html>
```

Then people can just use:
```
https://YOUR_USERNAME.github.io/Japanese_FastSpeech2/
```

### Option B: Use a URL shortener

- Use [bit.ly](https://bitly.com) or [tinyurl.com](https://tinyurl.com)
- Create a short link like: `bit.ly/tts-survey`

---

## ✅ Verify It Works

After deployment:
1. Visit your GitHub Pages URL
2. Check that audio files load and play
3. Complete a test survey
4. Verify you receive the email with results
5. Check that CSV/JSON files download

---

## 🔒 Important Notes

### Privacy:
- Your repository must be **public** for GitHub Pages (or you need GitHub Pro for private repos)
- Audio files will be publicly accessible
- Email address in the HTML is visible in source code (but that's okay - FormSubmit protects you from spam)

### File Size:
- GitHub has a 100MB file size limit per file
- Repository should be under 1GB total
- Your audio files should be fine (check with `du -sh voice_splits/`)

### Custom Domain (Optional):
- You can use your own domain name (e.g., `survey.mydomain.com`)
- Add a `CNAME` file to the repository
- Configure DNS settings with your domain provider

---

## 🐛 Troubleshooting

**"404 - Page not found"**
- Wait a few minutes for deployment
- Check that files are pushed to GitHub
- Verify the URL path is correct

**Audio files not loading:**
- Check browser console (F12) for errors
- Verify file paths are correct and files are committed
- GitHub Pages can take a few minutes to update

**Survey works locally but not on GitHub Pages:**
- Check that all file paths are relative (not absolute)
- Make sure all audio files are committed (not in .gitignore)
- Clear browser cache and try again

---

## 📊 Monitoring Responses

### Check Response Rate:
- GitHub Pages provides basic analytics in Settings > Pages
- Use the downloaded JSON/CSV files to analyze
- Check your email for submissions

### Collecting from Multiple People:
- Each response gets a unique Participant ID
- Timestamp shows when submitted
- Merge multiple CSV files if needed using a spreadsheet

---

## 🎯 Sharing the Survey

Once live, share the link via:
- ✉️ Email
- 💬 Social media
- 📱 QR code (generate at [qr-code-generator.com](https://www.qr-code-generator.com))
- 📄 Print/PDF with the link

Example message:
```
Please help with my research by rating Japanese TTS audio samples!
Takes about 10-15 minutes.

Link: https://YOUR_USERNAME.github.io/Japanese_FastSpeech2/voice_splits/audio_survey_email.html

Thank you! 🙏
```
