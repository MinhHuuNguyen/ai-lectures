# Cannot authenticate through git-credential as no helper is defined on your machine.
# You might have to re-authenticate when pushing to the Hugging Face Hub.
# Run the following command in your terminal in case you want to set the 'store' credential helper as default.
git config --global credential.helper store

# Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.

# To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens
uvx hf auth login
# Add token as git credential? [y/N]: y

cat /Users/minhhuunguyen/.cache/huggingface/stored_tokens
cat /Users/minhhuunguyen/.cache/huggingface/token
