#!/bin/bash
# run_quick_submission.sh
# Run the simple pipeline for Numerai Crypto for quick submissions

# Default parameters
TOURNAMENT="crypto"
TIME_BUDGET=30.0  # 30 minutes
SUBMIT_RESULTS=false
SUBMISSION_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tournament)
      TOURNAMENT="$2"
      shift 2
      ;;
    --time-budget)
      TIME_BUDGET="$2"
      shift 2
      ;;
    --submit)
      SUBMIT_RESULTS=true
      shift
      ;;
    --submission-id)
      SUBMISSION_ID="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --tournament TOURNAMENT   Tournament name (default: crypto)"
      echo "  --time-budget MINUTES     Time budget in minutes (default: 30.0)"
      echo "  --submission-id ID        Custom submission ID"
      echo "  --submit                  Submit results to Numerai API"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Display execution parameters
echo "Running quick submission pipeline with the following settings:"
echo "  Tournament:    $TOURNAMENT"
echo "  Time budget:   $TIME_BUDGET minutes"
echo "  Submit:        $SUBMIT_RESULTS"
if [ -n "$SUBMISSION_ID" ]; then
  echo "  Submission ID: $SUBMISSION_ID"
fi
echo

# Prepare command
COMMAND="python scripts/run_simple.py --tournament $TOURNAMENT --time-budget $TIME_BUDGET"

if [ -n "$SUBMISSION_ID" ]; then
  COMMAND="$COMMAND --submission-id '$SUBMISSION_ID'"
fi

if [ "$SUBMIT_RESULTS" = true ]; then
  COMMAND="$COMMAND --submit"
fi

# Run the command
echo "Executing: $COMMAND"
echo "---------------------------------------"
eval $COMMAND

# Check exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "---------------------------------------"
  echo "Quick submission pipeline completed successfully!"
else
  echo "---------------------------------------"
  echo "Quick submission pipeline failed with exit code $STATUS"
fi

exit $STATUS