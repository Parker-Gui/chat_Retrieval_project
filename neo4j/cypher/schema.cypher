CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (entity:Entity) REQUIRE entity.name IS UNIQUE;
