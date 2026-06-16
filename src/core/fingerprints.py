from __future__ import annotations

import hashlib

from scrum_master_assistant.models.issues import Issue, JiraStory


def issue_fingerprint(issue: Issue) -> str:
    payload = "|".join(
        [
            issue.category.value,
            issue.sub_category or "",
            issue.severity.value,
            issue.source_system,
            issue.service_name or "",
            issue.environment or "",
            issue.title,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def story_fingerprint(story: JiraStory) -> str:
    payload = "|".join(
        [
            story.project_key,
            story.epic_key,
            story.summary,
            story.priority,
            ",".join(sorted(story.labels)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
