#!/bin/bash
# run_optimal_pipeline.sh
# Run the optimal pipeline for Numerai Crypto with high memory and GPU acceleration

# Default parameters
TOURNAMENT="crypto"
TIME_BUDGET=8.0  # 8 hours
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
      echo "  --time-budget HOURS       Time budget in hours (default: 8.0)"
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
echo "Running optimal pipeline with the following settings:"
echo "  Tournament:    $TOURNAMENT"
echo "  Time budget:   $TIME_BUDGET hours"
echo "  Submit:        $SUBMIT_RESULTS"
if [ -n "$SUBMISSION_ID" ]; then
  echo "  Submission ID: $SUBMISSION_ID"
fi
echo

# Prepare command
COMMAND="python scripts/run_optimal.py --tournament $TOURNAMENT --time-budget $TIME_BUDGET"

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
  echo "Optimal pipeline completed successfully!"
else
  echo "---------------------------------------"
  echo "Optimal pipeline failed with exit code $STATUS"
fi

exit $STATUS