import requests

# API URLs
GET_LOCATIONS_URL = "https://dev-user-api.eruvaka.dev/1.0/users/locations"
DELETE_LOCATION_URL = "https://dev-user-api.eruvaka.dev/1.0/users/locations/{}"

# Fill in headers from your browser's network tab
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1M2Y1NmU1MTEwNTNjNTcwYTEyMTk3ZSIsImVtYWlsIjoicG9uZGxvZ3MucWFAZ21haWwuY29tIiwidXNlcl90eXBlIjoiQURNSU4iLCJmaXJzdF9uYW1lIjoiUG9uZGxvZ3MiLCJsYXN0X25hbWUiOiJRIEEgQWNjb3VudCIsInNvdXJjZSI6ImhvbWVwYWdlIiwidXNlcklkIjoiNjUzZjU2ZTUxMTA1M2M1NzBhMTIxOTdlIiwibWV0YV9kYXRhIjp7ImJyb3dzZXIiOiJNb3ppbGxhLzUuMCAoV2luZG93cyBOVCAxMC4wOyBXaW42NDsgeDY0KSBBcHBsZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSBDaHJvbWUvMTM5LjAuMC4wIFNhZmFyaS81MzcuMzYiLCJjbGllbnQiOiJXRUIifSwiaV91c2VyX2lkIjoiNWRmNzMyNmZhYjFhNGIzODc4ZDZjNjAwIiwiaV91c2VyX3R5cGUiOiJTVVBFUl9BRE1JTiIsImlhdCI6MTc1NjEwMTI1MCwiZXhwIjoxNzU2NzA2MDUwfQ.p-FsVN7vHQGUYtW7jRePhVDos82VfW4eNUNuv77-SbI",  # If shown in your requests
    "Content-Type": "application/json",
    "Cookie": "_csrf=g4q0gnSqQYTvZiPGQVI2FdAj; jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjVkZjczMjZmYWIxYTRiMzg3OGQ2YzYwMCIsImVtYWlsIjoic3VwZXIuYWRtaW4uMDFAZXJ1dmFrYS5jb20iLCJ1c2VyX3R5cGUiOiJTVVBFUl9BRE1JTiIsImZpcnN0X25hbWUiOiJTdXBlckFkbWluIGF0dCIsImxhc3RfbmFtZSI6IkVydXZha2EiLCJ1c2VySWQiOiI1ZGY3MzI2ZmFiMWE0YjM4NzhkNmM2MDAiLCJtZXRhX2RhdGEiOnsiYnJvd3NlciI6Ik1vemlsbGEvNS4wIChXaW5kb3dzIE5UIDEwLjA7IFdpbjY0OyB4NjQpIEFwcGxlV2ViS2l0LzUzNy4zNiAoS0hUTUwsIGxpa2UgR2Vja28pIENocm9tZS8xMzkuMC4wLjAgU2FmYXJpLzUzNy4zNiIsImNsaWVudCI6IldFQiJ9LCJpYXQiOjE3NTYxMDEyMTgsImV4cCI6MTc1NjcwNjAxOH0.YCWnXYivnMZc8lFpGUeZZTpmSTO09GxNjvzxbrOY-vI; XSRF-TOKEN=9msMbRWi-3tOjJYWmmu9XIGGoLEkG7if1HnI; mp_75d8126e3046b75d81e91422e47487c7_mixpanel=%7B%22distinct_id%22%3A%20%22653f56e511053c570a12197e%22%2C%22%24device_id%22%3A%20%22198dfd5a09c935-035e3a5c42b6b38-26011051-144000-198dfd5a09d1c54%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%2C%22%24user_id%22%3A%20%22653f56e511053c570a12197e%22%2C%22email%22%3A%20%22653f56e511053c570a12197e%22%2C%22fullName%22%3A%20%22Pondlogs%20Q%20A%20Account%22%7D",        # Only if present in your requests
    # Add any other headers from the network tab here
}

# Step 1: Get all locations
resp = requests.get(GET_LOCATIONS_URL, headers=headers)
resp.raise_for_status()
locations = resp.json()

# Step 2: Loop and delete all locations with name "A. Kothapalle"
for loc in locations:
    if loc.get("name") == "a. kothapalle":
        loc_id = loc.get("id")
        del_url = DELETE_LOCATION_URL.format(loc_id)
        del_resp = requests.delete(del_url, headers=headers)
        print(f"Deleted {loc_id}: Status {del_resp.status_code}")

# Optional: Confirm deletion
new_locations = requests.get(GET_LOCATIONS_URL, headers=headers).json()
remaining = [loc for loc in new_locations if loc.get("name") == "a. kothapalle"]
print(f"Remaining 'A. Kothapalle' locations: {len(remaining)}")
