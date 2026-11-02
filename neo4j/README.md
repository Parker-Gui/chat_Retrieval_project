# MovieGraph Neo4j

This project uses Neo4j 5.26.32 Community LTS. The graph contains one
`Entity` node per unique MetaQA entity and one relationship per unique knowledge
triple. MetaQA relation names are converted to uppercase Neo4j relationship types.

## One-command Windows setup

The default setup downloads the official Windows ZIP and does not require Docker.
It requires JDK 17 and the project Python 3.11 environment.

```powershell
.\scripts\neo4j\setup_neo4j.ps1
```

The script starts Neo4j, imports the processed MetaQA triples, and verifies graph
counts and sample 1-hop, 2-hop, and 3-hop paths.

Neo4j Browser is available at http://localhost:7474. Local development credentials
are configured in `neo4j/.env`, which is excluded from Git.

## Start and stop

```powershell
.\scripts\neo4j\start_neo4j.ps1
.\scripts\neo4j\stop_neo4j.ps1
```

Stopping native Neo4j preserves the graph under the ignored local `tools/` directory.
To inspect useful queries, open `neo4j/cypher/examples.cypher`.

## Optional Docker setup

If Docker Desktop is working, the equivalent Docker command is:

```powershell
.\scripts\neo4j\setup_neo4j_docker.ps1
```

The Docker deployment uses the pinned `neo4j:5.26.32-community` image and named
volumes for persistence.
