# Copilot 向け指示

- `OPENAI_API_KEY` が必要なコマンドを実行する前に、`zsh -lc 'source ~/.zshrc && <command>'` を使って、ユーザーのシェル設定からキーを読み込むこと。
- Bash の中で直接 `source ~/.zshrc` しないこと。`.zshrc` には `autoload` や `compinit` のような Zsh 専用設定が含まれている場合がある。
- Pull Request と Issue は日本語で作成すること。
