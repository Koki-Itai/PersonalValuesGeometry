source .env

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set in .env file"
    exit 1
fi

huggingface-cli login --token "$HUGGINGFACE_TOKEN"