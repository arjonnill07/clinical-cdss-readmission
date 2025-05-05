#!/bin/sh

# This script injects environment variables into the frontend HTML
# It's used in the Docker container at runtime

# Replace the placeholder with the actual API URL
if [ -n "$API_URL" ]; then
  echo "Injecting API_URL: $API_URL"
  sed -i "s|window.API_URL || 'http://localhost:8000'|'$API_URL'|g" /usr/share/nginx/html/index.html
else
  echo "API_URL not set, using default"
fi

# Execute the CMD from the Dockerfile
exec "$@"
