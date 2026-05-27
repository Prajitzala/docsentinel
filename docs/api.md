# Parser API

## parse_repo

`parse_repo(repo_path)` walks the repository and returns code chunks and documentation sections.

It parses Python files with the AST and splits Markdown files by heading.

The optional `verbose` flag is reserved for future progress logging.
