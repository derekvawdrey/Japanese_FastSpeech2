# TTS Audio Quality Survey

This folder contains survey tools to collect ratings on Japanese TTS audio samples.

## 📁 Files

### Survey Options (Choose One):

1. **`audio_survey_email.html`** ⭐ **RECOMMENDED FOR BEGINNERS**
   - ✅ Easiest setup (1 minute)
   - 📧 Results sent to your email automatically
   - 💾 Also downloads CSV/JSON backups
   - 🆓 Completely free (uses FormSubmit.co)
   
   **Setup:** Just change one line with your email address!

2. **`audio_survey.html`** 
   - 📊 Saves to Google Sheets (requires 5 min setup)
   - 📈 Best for organized data analysis
   - 🔄 Real-time updates in spreadsheet
   - 💾 Also downloads CSV/JSON backups
   
   **Setup:** Follow instructions in `SETUP_INSTRUCTIONS.md`

3. **Original `audio_survey.html`** (without configuration)
   - 💾 Local downloads only (CSV + JSON)
   - ⚡ Zero setup required
   - 🔒 Most private (no cloud storage)
   - 📂 You collect files manually from participants

### Documentation:

- **`SETUP_INSTRUCTIONS.md`** - Detailed setup guide for all options
- **`sample_text.txt`** - List of Japanese sentences used in survey

## 🚀 Quick Start

### For Email Collection (Recommended):

1. Open `audio_survey_email.html` in a text editor
2. Find: `const YOUR_EMAIL = 'your.email@example.com';`
3. Replace with your real email
4. Save and open in a web browser
5. Done! Share with participants

### For Google Sheets:

1. Follow the detailed guide in `SETUP_INSTRUCTIONS.md`
2. Create a Google Apps Script (5 minutes)
3. Configure the survey HTML with your script URL
4. Open in browser and test

### For Local Only:

1. Open `audio_survey.html` in a browser
2. That's it! Results will download when submitted

## 📊 Survey Details

- **Total Samples:** 60 (15 sentences × 4 audio sources)
- **Rating Scale:** 1-5 (1 = Robotic 🤖, 5 = Human 👤)
- **Audio Sources:**
  - Mine (WAV)
  - Sample (WAV)
  - Google Translate (MP3)
  - Ondoku (MP3)

## 🎯 What Gets Saved

Each submission includes:
- Timestamp
- Unique Participant ID
- Individual ratings for all 60 samples
- Summary statistics (average, min, max per source)
- Participant email (if using email version)

## 💡 Tips

- **Test first!** Complete a test survey yourself before sharing
- **Email version:** First submission requires email confirmation (check spam!)
- **Google Sheets:** View real-time results as people submit
- **Multiple participants:** Each gets a unique ID (P + timestamp)
- **Privacy:** Email is optional (except in email version for delivery)

## 🔧 Troubleshooting

**Email not arriving?**
- Check spam/junk folder
- Verify email address in the HTML file is correct
- First submission requires activation (check email for confirmation link)

**Google Sheets not updating?**
- Double-check the script URL in the HTML
- Make sure deployment is set to "Anyone" can access
- Check browser console (F12) for error messages

**Audio not playing?**
- Ensure all audio files are in correct folders (mine/, sample/, etc.)
- Check that file names match exactly (including Japanese characters)
- Try a different web browser

## 📫 Need Help?

See `SETUP_INSTRUCTIONS.md` for detailed troubleshooting and advanced options.
