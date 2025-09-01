import requests

# Endpoints
LIST_URL = "https://dev-user-api.eruvaka.dev/1.0/users/locations"
DELETE_URL = "https://dev-user-api.eruvaka.dev/1.0/users/locations/{}"

headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1M2Y1NmU1MTEwNTNjNTcwYTEyMTk3ZSIsImVtYWlsIjoicG9uZGxvZ3MucWFAZ21haWwuY29tIiwidXNlcl90eXBlIjoiQURNSU4iLCJmaXJzdF9uYW1lIjoiUG9uZGxvZ3MiLCJsYXN0X25hbWUiOiJRIEEgQWNjb3VudCIsInNvdXJjZSI6ImhvbWVwYWdlIiwidXNlcklkIjoiNjUzZjU2ZTUxMTA1M2M1NzBhMTIxOTdlIiwibWV0YV9kYXRhIjp7ImJyb3dzZXIiOiJNb3ppbGxhLzUuMCAoV2luZG93cyBOVCAxMC4wOyBXaW42NDsgeDY0KSBBcHBsZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSBDaHJvbWUvMTM5LjAuMC4wIFNhZmFyaS81MzcuMzYiLCJjbGllbnQiOiJXRUIifSwiaV91c2VyX2lkIjoiNWRmNzMyNmZhYjFhNGIzODc4ZDZjNjAwIiwiaV91c2VyX3R5cGUiOiJTVVBFUl9BRE1JTiIsImlhdCI6MTc1NjEwMTI1MCwiZXhwIjoxNzU2NzA2MDUwfQ.p-FsVN7vHQGUYtW7jRePhVDos82VfW4eNUNuv77-SbI",
    "origin": "https://pondlogs.eruvaka.dev",
    "priority": "u=1, i",
    "referer": "https://pondlogs.eruvaka.dev/",
    "sec-ch-ua": "\"Not;A=Brand\";v=\"99\", \"Google Chrome\";v=\"139\", \"Chromium\";v=\"139\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "xsrf-token": "9msMbRWi-3tOjJYWmmu9XIGGoLEkG7if1HnI",
}

cookies = {
    "_csrf": "g4q0gnSqQYTvZiPGQVI2FdAj",
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjVkZjczMjZmYWIxYTRiMzg3OGQ2YzYwMCIsImVtYWlsIjoic3VwZXIuYWRtaW4uMDFAZXJ1dmFrYS5jb20iLCJ1c2VyX3R5cGUiOiJTVVBFUl9BRE1JTiIsImZpcnN0X25hbWUiOiJTdXBlckFkbWluIGF0dCIsImxhc3RfbmFtZSI6IkVydXZha2EiLCJ1c2VySWQiOiI1ZGY3MzI2ZmFiMWE0YjM4NzhkNmM2MDAiLCJtZXRhX2RhdGEiOnsiYnJvd3NlciI6Ik1vemlsbGEvNS4wIChXaW5kb3dzIE5UIDEwLjA7IFdpbjY0OyB4NjQpIEFwcGxlV2ViS2l0LzUzNy4zNiAoS0hUTUwsIGxpa2UgR2Vja28pIENocm9tZS8xMzkuMC4wLjAgU2FmYXJpLzUzNy4zNiIsImNsaWVudCI6IldFQiJ9LCJpYXQiOjE3NTYxMDEyMTgsImV4cCI6MTc1NjcwNjAxOH0.YCWnXYivnMZc8lFpGUeZZTpmSTO09GxNjvzxbrOY-vI",
    "XSRF-TOKEN": "9msMbRWi-3tOjJYWmmu9XIGGoLEkG7if1HnI",
    "mp_75d8126e3046b75d81e91422e47487c7_mixpanel": "{\"distinct_id\":\"653f56e511053c570a12197e\",\"$device_id\":\"198dfd5a09c935-035e3a5c42b6b38-26011051-144000-198dfd5a09d1c54\",\"$initial_referrer\":\"$direct\",\"$initial_referring_domain\":\"$direct\",\"$user_id\":\"653f56e511053c570a12197e\",\"email\":\"653f56e511053c570a12197e\",\"fullName\":\"Pondlogs Q A Account\"}"
}

# Step 1: List locations and find the first named 'A. Kothapalle'
response = requests.get(LIST_URL, headers=headers, cookies=cookies)
response.raise_for_status()
locations = response.json()

found = False
for loc in locations["user_locations"]:
    print(loc)
    if loc.get("name") == "a. kothapalle":
        loc_id = loc.get("_id")
        print(f"Found: name='a. kothapalle', id={loc_id}")
        # Step 2: Delete this location
        del_url = DELETE_URL.format(loc_id)
        del_response = requests.delete(del_url, headers=headers, cookies=cookies)
        print(f"Deleted {loc_id}: HTTP {del_response.status_code}")
        print(f"Response: {del_response.text}")
        found = True

if not found:
    print("No location with name 'A. Kothapalle' found.")
