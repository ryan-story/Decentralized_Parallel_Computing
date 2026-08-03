#!/usr/bin/env bash
# Launch the Chapter 2 recommender and open it on localhost.
#
#   cd app && ./app_launch.sh
#
# The first run builds the model across the simulated workers and caches it, so
# it takes a second or two; after that the app serves queries in milliseconds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "streamlit is not installed. Install the chapter's requirements first:"
  echo "    pip install -r ../requirements.txt"
  exit 1
fi

echo "Starting the recommender on http://localhost:8501"
exec python -m streamlit run "$HERE/app.py" --server.port 8501 "$@"
