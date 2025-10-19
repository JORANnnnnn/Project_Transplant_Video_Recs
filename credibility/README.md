# Credibility Assessment

**Author:** Zihan Liu
**Purpose:** Verify that collected YouTube videos are from credible, verified organizations

## Folder Structure

```
credibility/
├── data/
│   └── Transplants_Team_Master_File.xlsx    # Master list of verified organizations
├── output/
│   ├── All_YouTube_Channels_Collected.xlsx  # All channels from video database
│   ├── Unmatched_Videos_For_Review.csv      # Videos from non-verified sources
│   └── Credibility_Hit_Rate_Report.xlsx     # Full hit rate analysis report
├── credibility_hit_rate_analysis.ipynb      # Main analysis notebook
├── create_verified_orgs_db.py               # Database creation script
├── verified_organizations.db                 # Generated database
└── README.md
```

## Quick Start

1. Ensure you have `youtube_data.db` in the parent folder
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook credibility_hit_rate_analysis.ipynb
   ```
3. Check the `output/` folder for results

## Files

### Input
- `data/Transplants_Team_Master_File.xlsx` - list of 273 verified organizations with YouTube handles

### Output
- `verified_organizations.db` - SQLite database of verified organizations
- `output/All_YouTube_Channels_Collected.xlsx` - All channels from video database
- `output/Unmatched_Videos_For_Review.csv` - Videos from non-verified channels
- `output/Credibility_Hit_Rate_Report.xlsx` - Complete hit rate analysis

## Database Schema

**Table: verified_organizations**

| Column | Type | Description |
|--------|------|-------------|
| organization | TEXT | Organization name |
| youtube_channel | TEXT | YouTube handle |
| rank | INTEGER | Credibility/volume rank (NULL for now) |

## Hit Rate Metrics

### Channel Hit Rate
Percentage of YouTube channels that match verified organizations
```
Channel Hit Rate = (Matched Channels / Total Channels) × 100
```

### Video Hit Rate
Percentage of videos from verified organizations
```
Video Hit Rate = (Videos from Verified Channels / Total Videos) × 100
```

## Matching Logic

Videos are matched to verified organizations using YouTube handles:
1. Normalize handles (remove @, lowercase)
2. Exact match on handle: `@uwhealth` = `@uwhealth`
3. Videos from matched channels are marked as verified

## SQL Query Examples

### Get all verified organizations
```sql
SELECT * FROM verified_organizations;
```

### Get organizations with YouTube channels
```sql
SELECT * FROM verified_organizations
WHERE youtube_channel IS NOT NULL;
```

### Check if a YouTube channel is verified
```sql
SELECT * FROM verified_organizations
WHERE youtube_channel = '@DaVitaKidneyCare';
```

### Get videos from verified organizations
```sql
SELECT v.*, vo.organization, vo.rank
FROM Videos v
JOIN Channels c ON v.channel_id = c.channel_id
JOIN verified_organizations vo ON
    REPLACE(LOWER(c.handle), '@', '') = REPLACE(LOWER(vo.youtube_channel), '@', '')
WHERE vo.youtube_channel IS NOT NULL;
```

## Adding New Organizations

1. Add to `data/Transplants_Team_Master_File.xlsx`
2. Re-run the notebook (database will auto-update)
3. Check updated hit rates

## Requirements

```
pandas
openpyxl
matplotlib
sqlite3
```

## Notes

- The notebook automatically creates/updates the database from the master Excel file
- Handles are case-insensitive and @ symbols are normalized
- The master file should be updated when new credible organizations are identified
