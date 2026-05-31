# Copilot instructions

- Before running commands that need `OPENAI_API_KEY`, use `zsh -lc 'source ~/.zshrc && <command>'` so the key is loaded from the user's shell config.
- Avoid `source ~/.zshrc` directly inside Bash; `.zshrc` may contain Zsh-only setup like `autoload` and `compinit`.
- Create pull requests and issues in Japanese.
