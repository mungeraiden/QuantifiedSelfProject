import pandas as pd

dates = pd.date_range(start="2025-08-14", end="2026-05-13", freq="D")
cal = pd.DataFrame({"date": dates})

def mark_range(df, start, end, col, value=1, name=None):
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    df.loc[mask, col] = value
    if name:
        df.loc[mask, "event_name"] = name

cal["event_name"] = ""
cal["is_holiday"] = 0
cal["is_break"] = 0
cal["is_school_day"] = 1
cal["semester"] = ""

mark_range(cal, "2025-08-14", "2025-08-15", "is_school_day", 0, "New Faculty Orientation")
mark_range(cal, "2025-08-21", "2025-08-21", "is_school_day", 0, "Fall Faculty Conference / LeAD")
mark_range(cal, "2025-08-22", "2025-08-22", "is_school_day", 0, "Reserved for Faculty Meetings")
mark_range(cal, "2025-08-22", "2025-08-25", "is_school_day", 0, "Welcome Weekend")

mark_range(cal, "2025-08-26", "2025-12-12", "semester", "Fall 2025")

mark_range(cal, "2025-09-01", "2025-09-01", "is_holiday", 1, "Labor Day")
mark_range(cal, "2025-10-20", "2025-10-20", "is_holiday", 1, "Founder's Day")
mark_range(cal, "2025-11-26", "2025-11-28", "is_break", 1, "Thanksgiving Break")
mark_range(cal, "2025-12-15", "2025-12-31", "is_break", 1, "Christmas Break")

mark_range(cal, "2026-01-11", "2026-01-11", "is_school_day", 0, "Residence Halls Open")
mark_range(cal, "2026-01-12", "2026-01-12", "is_school_day", 0, "Spring Faculty Conference / Advising Summit")

mark_range(cal, "2026-01-13", "2026-05-08", "semester", "Spring 2026")

mark_range(cal, "2026-01-19", "2026-01-19", "is_holiday", 1, "MLK Day")
mark_range(cal, "2026-02-16", "2026-02-16", "is_holiday", 1, "Presidents' Day")
mark_range(cal, "2026-03-09", "2026-03-13", "is_break", 1, "Spring Break")
mark_range(cal, "2026-04-03", "2026-04-03", "is_holiday", 1, "Good Friday")
mark_range(cal, "2026-04-06", "2026-04-06", "is_holiday", 1, "Easter Monday")

mark_range(cal, "2026-05-02", "2026-05-04", "is_break", 1, "Reading/Study Days")

cal.loc[(cal["is_holiday"] == 1) | (cal["is_break"] == 1), "is_school_day"] = 0

cal["semester"] = cal["semester"].replace("", None).ffill()

cal.to_csv("academic_calendar.csv", index=False)

print("academic_calendar.csv created successfully!")
print(cal.head(20))
