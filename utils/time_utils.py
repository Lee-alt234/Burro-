from datetime import datetime, time
import re

def is_place_open_now(timings: list) -> tuple[bool, str]:
    now = datetime.now()
    today = now.strftime("%A")

    # Handle if timings is None or not a list
    if not timings or not isinstance(timings, list):
        return (False, "Timing information not available.")

    # Find today's timing
    today_timing = next((t for t in timings if today.lower() in t.lower()), None)
    if not today_timing or "closed" in today_timing.lower():
        return (False, f"{today} is a closed day.")

    # Extract opening and closing times using regex
    match = re.findall(r"(\d{1,2}:\d{2}\s?[APMapm]+)", today_timing)
    if len(match) < 2:
        return (False, "Invalid time format.")

    try:
        open_time = datetime.strptime(match[0], "%I:%M %p").time()
        close_time = datetime.strptime(match[1], "%I:%M %p").time()
        current_time = now.time()

        # Handle past-midnight closing (e.g., 10 AM – 1 AM next day)
        if open_time < close_time:
            is_open = open_time <= current_time <= close_time
        else:
            is_open = current_time >= open_time or current_time <= close_time

        if is_open:
            return (True, f"Open now until {close_time.strftime('%I:%M %p')}")
        else:
            return (False, f"Closed now — opens at {open_time.strftime('%I:%M %p')}")

    except Exception as e:
        return (False, f"Error parsing time: {e}")
