// Overall graph statistics
MATCH (entity:Entity)
WITH count(entity) AS entities
MATCH ()-[relationship]->()
RETURN entities, count(relationship) AS relationships;

// Count each MetaQA relationship type
MATCH ()-[relationship]->()
RETURN type(relationship) AS relationship_type, count(*) AS count
ORDER BY relationship_type;

// 1-hop example: Who directed Inception?
MATCH (:Entity {name: "Inception"})-[:DIRECTED_BY]->(director:Entity)
RETURN director.name AS answer;

// 2-hop example: Movies by the director of Inception
MATCH (:Entity {name: "Inception"})-[:DIRECTED_BY]->(director:Entity)
      <-[:DIRECTED_BY]-(movie:Entity)
RETURN DISTINCT movie.name AS answer
ORDER BY answer;

// Generic graph neighborhood for visual inspection
MATCH path=(entity:Entity {name: "Inception"})-[*1..2]-(related:Entity)
RETURN path
LIMIT 50;
