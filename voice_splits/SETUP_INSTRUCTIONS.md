# Survey Database Setup Instructions

## Quick Comparison

| Method | Setup Time | Ease of Use | Data Format | Best For |
|--------|------------|-------------|-------------|----------|
| **Email (FormSubmit)** | 1 min | ⭐⭐⭐⭐⭐ Easiest | Email + CSV/JSON | Quick surveys, few responses |
| **Google Sheets** | 5 min | ⭐⭐⭐⭐ Easy | Spreadsheet | Organized data, easy analysis |
| **Local Only** | 0 min | ⭐⭐⭐⭐⭐ Instant | CSV/JSON download | Testing, no internet needed |

---

## Option 1: Email Collection (Easiest - 1 Minute Setup!)

### File: `audio_survey_email.html`

**No account needed! Results sent to your email automatically.**

### Setup:
1. Open `audio_survey_email.html` in a text editor
2. Find this line near the top:
   ```javascript
   const YOUR_EMAIL = 'your.email@example.com';
   ```
3. Replace `your.email@example.com` with your actual email
4. Save and open in browser - Done! ✅

### How it works:
- Participants fill out the survey
- Results are automatically emailed to you as JSON
- Also downloads CSV/JSON files as backup
- Uses [FormSubmit.co](https://formsubmit.co/) (free service)

### First-time activation:
- The first submission will send you a confirmation email
- Click the link to activate
- After that, all responses come through automatically!

---

## Option 2: Google Sheets (Best for Analysis - 5 Minute Setup)

### Step 1: Create a Google Sheet
1. Go to [Google Sheets](https://sheets.google.com)
2. Create a new spreadsheet
3. Name it "TTS Survey Results"
4. In the first row, add these headers:
   - A1: `Timestamp`
   - B1: `Participant ID`
   - C1: `Sentence`
   - D1: `Source`
   - E1: `Rating`

### Step 2: Create the Apps Script
1. In your Google Sheet, click **Extensions** → **Apps Script**
2. Delete any existing code
3. Copy and paste this code:

```javascript
function doPost(e) {
  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = JSON.parse(e.postData.contents);
    
    // Add each rating as a row
    for (let [key, rating] of Object.entries(data.ratings)) {
      const parts = key.split('_');
      const source = parts.pop();
      const sentence = parts.join('_');
      
      sheet.appendRow([
        data.timestamp,
        data.participantId,
        sentence,
        source,
        rating
      ]);
    }
    
    return ContentService.createTextOutput(JSON.stringify({
      status: 'success',
      message: 'Data saved successfully'
    })).setMimeType(ContentService.MimeType.JSON);
    
  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({
      status: 'error',
      message: error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}
```

4. Click **Save** (disk icon)
5. Click **Deploy** → **New deployment**
6. Click the gear icon ⚙️ next to "Select type"
7. Select **Web app**
8. Configure:
   - Description: "TTS Survey Webhook"
   - Execute as: **Me**
   - Who has access: **Anyone**
9. Click **Deploy**
10. **Copy the Web app URL** (it looks like: `https://script.google.com/macros/s/ABC.../exec`)
11. Click **Done**

### Step 3: Update the Survey HTML
1. Open `audio_survey.html`
2. Find the line near the top that says: `const GOOGLE_SCRIPT_URL = 'YOUR_SCRIPT_URL_HERE';`
3. Replace `YOUR_SCRIPT_URL_HERE` with your Web app URL
4. Save the file

### Step 4: Test It!
1. Open the survey in a browser
2. Rate some samples
3. Submit the survey
4. Check your Google Sheet - the data should appear!

---

## Option 3: Local Only (No Setup - Instant!)

### File: Original `audio_survey.html` (without Google Script URL configured)

**Just open and use - no configuration needed!**

- Results download automatically as CSV and JSON files
- No internet connection required for saving
- Perfect for testing or small-scale surveys
- You manually collect the downloaded files from participants

---

## Advanced Option: Firebase Realtime Database

If you need real-time updates or want more control:

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project (free tier)
3. Add a Web app
4. Enable Realtime Database
5. Set security rules to allow writes
6. Copy the Firebase config
7. Let me know if you want the code for this!

---

## Troubleshooting

**Issue**: "Authorization required" error
- **Solution**: Make sure "Who has access" is set to "Anyone" in the deployment settings

**Issue**: Data not appearing in the sheet
- **Solution**: Check the browser console (F12) for errors. Make sure the script URL is correct.

**Issue**: CORS error
- **Solution**: This shouldn't happen with Google Apps Script, but if it does, try redeploying the script.
