# create a docker network for morpheus
docker network inspect morpheus >/dev/null 2>&1 || docker network create morpheus

# create the parent conda folder so it's found when mounting
mkdir -p ./.cache/conda
