# Postman verification

Import the collection, set base_url to the hosted API base URL and api_key to your private API key, then select each file from docs/samples. Run the requests in order. The history assertion checks that the last prediction was saved. The last three requests intentionally expect 415, 400 and 413.

Newman can execute the same collection with the project root as its working directory. Keep any environment containing the real API key out of Git and screenshots.
