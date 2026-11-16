"""Loss-aware embedded vector store migration."""

__version__ = "0.1.0"

# Listed by ``ragwrench migrate --help`` when the chroma extra is missing.
SUBCOMMANDS = {
    "copy": "copy a Chroma or Qdrant local collection into the other store",
    "verify": "check record and top-k parity between two stores",
    "inspect": "describe a local Chroma or Qdrant collection",
}
