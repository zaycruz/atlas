from kg_client import KG

kg = KG()

def retrieve_context(project_id: str, topic_name: str, layer: str | None = None, start_iso: str | None = None, end_iso: str | None = None, limit: int = 10):
    items = kg.find(
        type='Memory',
        projectId=project_id,
        topicName=topic_name,
        layer=layer,
        timeStart=start_iso,
        timeEnd=end_iso,
    )
    return items[:limit]
