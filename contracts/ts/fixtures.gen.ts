// Synthetic contract examples, never production analysis.
import type { CheckDetailResponse, JobDone, JobFailed, JobProcessing, ReevaluateResponse, RulesResponse } from './types.gen';

export const v1_done_violation: JobDone = {
  "job_id": "7c1e2a9b40d3",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Пересчёт денег без клиента у кассы",
      "timecode_sec": 44.2,
      "description": "Кассир держит деньги, место клиента у окна видно и свободно. Синтетический пример структуры ответа.",
      "evidence_frames": [
        {
          "timestamp_sec": 44.2,
          "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAAkAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5dooorU+1CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//9k="
        }
      ]
    },
    {
      "type": "Посторонний в рабочей зоне кассира",
      "timecode_sec": 92.0,
      "description": "Человек с неуточнённой ролью находится в рабочей зоне кассира. Синтетический пример структуры ответа.",
      "evidence_frames": [
        {
          "timestamp_sec": 92.0,
          "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAAkAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5dooorU+1CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//9k="
        }
      ]
    }
  ]
};

export const v1_done_clean: JobDone = {
  "job_id": "7c1e2a9b40d3",
  "status": "done",
  "verdict": "no_violation",
  "violations": []
};

export const v1_processing: JobProcessing = {
  "job_id": "7c1e2a9b40d3",
  "status": "processing"
};

export const v1_failed: JobFailed = {
  "job_id": "7c1e2a9b40d3",
  "status": "failed",
  "error": {
    "code": "VIDEO_DECODE_FAILED",
    "message": "Не удалось прочитать кадр."
  }
};

export const ui_check_processing: CheckDetailResponse = {
  "check": {
    "id": "7c1e2a9b40d3",
    "file_name": "synthetic-contract-example.mov",
    "status": "processing",
    "created_at": "2026-10-17T07:30:00Z",
    "duration_sec": 125.04,
    "verdict": null,
    "findings_count": null,
    "is_example": true,
    "analysis_origin": "local",
    "analyzed_at": null,
    "automatic_verdict": null,
    "video": {
      "duration_sec": 125.04,
      "fps_nominal": 29.97,
      "width": 1920,
      "height": 1080,
      "rotation": 0
    },
    "progress": {
      "phase": "perception",
      "phases": [
        "decode",
        "perception",
        "facts",
        "rules",
        "evidence"
      ],
      "processed_until_sec": 48.0,
      "queue_position": null
    },
    "layout": {
      "revision": 1,
      "zones": [
        {
          "zone_id": "z-cash_desk-1",
          "kind": "cash_desk",
          "polygon": [
            [
              0.1,
              0.4
            ],
            [
              0.7,
              0.4
            ],
            [
              0.7,
              0.9
            ],
            [
              0.1,
              0.9
            ]
          ],
          "source": "auto"
        },
        {
          "zone_id": "z-client_window-1",
          "kind": "client_window",
          "polygon": [
            [
              0.3,
              0.1
            ],
            [
              0.7,
              0.1
            ],
            [
              0.7,
              0.39
            ],
            [
              0.3,
              0.39
            ]
          ],
          "source": "auto"
        }
      ],
      "camera_note": null
    },
    "tracks": [
      {
        "track_id": "p1",
        "kind": "person",
        "object_class": null,
        "first_sec": 0.0,
        "last_sec": 125.0,
        "role": {
          "value": "cashier",
          "source": "auto"
        },
        "samples": []
      },
      {
        "track_id": "p2",
        "kind": "person",
        "object_class": null,
        "first_sec": 90.0,
        "last_sec": 96.0,
        "role": {
          "value": "unknown",
          "source": "auto"
        },
        "samples": []
      }
    ],
    "observations": [
      {
        "start_sec": 41.5,
        "end_sec": 47.5,
        "obs_id": "ob-1",
        "kind": "cash_in_hands",
        "subjects": [
          "p1",
          "o1"
        ],
        "zones": [
          "z-cash_desk-1"
        ],
        "score": 0.87,
        "frame_ids": [
          "fr-44200"
        ],
        "provenance": null,
        "model_id": null,
        "actor_id": null,
        "recipient_id": null
      }
    ],
    "coverage": [
      {
        "start_sec": 72.0,
        "end_sec": 80.0,
        "scope": "desk",
        "reason": "occluded"
      }
    ],
    "finding_set": null,
    "error": null
  }
};

export const ui_check_done: CheckDetailResponse = {
  "check": {
    "id": "7c1e2a9b40d3",
    "file_name": "synthetic-contract-example.mov",
    "status": "done",
    "created_at": "2026-10-17T07:30:00Z",
    "duration_sec": 125.04,
    "verdict": "violation",
    "findings_count": 2,
    "is_example": true,
    "analysis_origin": "local",
    "analyzed_at": null,
    "automatic_verdict": null,
    "video": {
      "duration_sec": 125.04,
      "fps_nominal": 29.97,
      "width": 1920,
      "height": 1080,
      "rotation": 0
    },
    "progress": {
      "phase": "done",
      "phases": [
        "decode",
        "perception",
        "facts",
        "rules",
        "evidence"
      ],
      "processed_until_sec": 125.04,
      "queue_position": null
    },
    "layout": {
      "revision": 1,
      "zones": [
        {
          "zone_id": "z-cash_desk-1",
          "kind": "cash_desk",
          "polygon": [
            [
              0.1,
              0.4
            ],
            [
              0.7,
              0.4
            ],
            [
              0.7,
              0.9
            ],
            [
              0.1,
              0.9
            ]
          ],
          "source": "auto"
        },
        {
          "zone_id": "z-client_window-1",
          "kind": "client_window",
          "polygon": [
            [
              0.3,
              0.1
            ],
            [
              0.7,
              0.1
            ],
            [
              0.7,
              0.39
            ],
            [
              0.3,
              0.39
            ]
          ],
          "source": "auto"
        }
      ],
      "camera_note": null
    },
    "tracks": [
      {
        "track_id": "p1",
        "kind": "person",
        "object_class": null,
        "first_sec": 0.0,
        "last_sec": 125.0,
        "role": {
          "value": "cashier",
          "source": "auto"
        },
        "samples": []
      },
      {
        "track_id": "p2",
        "kind": "person",
        "object_class": null,
        "first_sec": 90.0,
        "last_sec": 96.0,
        "role": {
          "value": "unknown",
          "source": "auto"
        },
        "samples": []
      }
    ],
    "observations": [
      {
        "start_sec": 41.5,
        "end_sec": 47.5,
        "obs_id": "ob-1",
        "kind": "cash_in_hands",
        "subjects": [
          "p1",
          "o1"
        ],
        "zones": [
          "z-cash_desk-1"
        ],
        "score": 0.87,
        "frame_ids": [
          "fr-44200"
        ],
        "provenance": null,
        "model_id": null,
        "actor_id": null,
        "recipient_id": null
      },
      {
        "start_sec": 90.0,
        "end_sec": 96.0,
        "obs_id": "ob-2",
        "kind": "person_in_zone",
        "subjects": [
          "p2"
        ],
        "zones": [
          "z-cashier_post-1"
        ],
        "score": 0.84,
        "frame_ids": [
          "fr-92000"
        ],
        "provenance": null,
        "model_id": null,
        "actor_id": null,
        "recipient_id": null
      }
    ],
    "coverage": [
      {
        "start_sec": 72.0,
        "end_sec": 80.0,
        "scope": "desk",
        "reason": "occluded"
      }
    ],
    "finding_set": {
      "revision": 1,
      "basis": "auto",
      "created_at": "2026-10-17T07:30:00Z",
      "layout_revision": 1,
      "role_overrides": [],
      "verdict": "violation",
      "findings": [
        {
          "finding_id": "fd-288d1b1423",
          "rule_id": "V-COUNT-ABSENT",
          "type_label": "Пересчёт денег без клиента у кассы",
          "timecode_sec": 44.2,
          "window": {
            "start_sec": 41.5,
            "end_sec": 47.5
          },
          "basis": [
            "ob-1"
          ],
          "what_seen": "Кассир держит деньги, место клиента у окна видно и свободно.",
          "why_may_violate": "Условие требует проверки человеком.",
          "norm_refs": [
            "S231-p20"
          ],
          "description": "Кассир держит деньги, место клиента у окна видно и свободно. Синтетический пример структуры ответа.",
          "evidence": [
            {
              "frame_id": "fr-44200",
              "timestamp_sec": 44.2,
              "overlay_boxes": []
            }
          ],
          "strength": "moderate",
          "decision": null
        },
        {
          "finding_id": "fd-3e26dc0700",
          "rule_id": "V-OUTSIDER",
          "type_label": "Посторонний в рабочей зоне кассира",
          "timecode_sec": 92.0,
          "window": {
            "start_sec": 90.0,
            "end_sec": 96.0
          },
          "basis": [
            "ob-2"
          ],
          "what_seen": "Человек с неуточнённой ролью находится в рабочей зоне кассира.",
          "why_may_violate": "Условие требует проверки человеком.",
          "norm_refs": [
            "S14-p15"
          ],
          "description": "Человек с неуточнённой ролью находится в рабочей зоне кассира. Синтетический пример структуры ответа.",
          "evidence": [
            {
              "frame_id": "fr-92000",
              "timestamp_sec": 92.0,
              "overlay_boxes": []
            }
          ],
          "strength": "moderate",
          "decision": null
        }
      ],
      "rules": [
        {
          "rule_id": "V-COUNT-ABSENT",
          "status": "partly_checkable",
          "not_checkable_windows": [
            {
              "start_sec": 72.0,
              "end_sec": 80.0,
              "reason": "Стол перекрыт."
            }
          ]
        },
        {
          "rule_id": "V-OUTSIDER",
          "status": "violation",
          "not_checkable_windows": []
        }
      ]
    },
    "error": null
  }
};

export const ui_check_failed: CheckDetailResponse = {
  "check": {
    "id": "7c1e2a9b40d3",
    "file_name": "synthetic-contract-example.mov",
    "status": "failed",
    "created_at": "2026-10-17T07:30:00Z",
    "duration_sec": 125.04,
    "verdict": null,
    "findings_count": null,
    "is_example": true,
    "analysis_origin": "local",
    "analyzed_at": null,
    "automatic_verdict": null,
    "video": {
      "duration_sec": 125.04,
      "fps_nominal": 29.97,
      "width": 1920,
      "height": 1080,
      "rotation": 0
    },
    "progress": {
      "phase": "failed",
      "phases": [
        "decode",
        "perception",
        "facts",
        "rules",
        "evidence"
      ],
      "processed_until_sec": 48.0,
      "queue_position": null
    },
    "layout": {
      "revision": 1,
      "zones": [
        {
          "zone_id": "z-cash_desk-1",
          "kind": "cash_desk",
          "polygon": [
            [
              0.1,
              0.4
            ],
            [
              0.7,
              0.4
            ],
            [
              0.7,
              0.9
            ],
            [
              0.1,
              0.9
            ]
          ],
          "source": "auto"
        },
        {
          "zone_id": "z-client_window-1",
          "kind": "client_window",
          "polygon": [
            [
              0.3,
              0.1
            ],
            [
              0.7,
              0.1
            ],
            [
              0.7,
              0.39
            ],
            [
              0.3,
              0.39
            ]
          ],
          "source": "auto"
        }
      ],
      "camera_note": null
    },
    "tracks": [
      {
        "track_id": "p1",
        "kind": "person",
        "object_class": null,
        "first_sec": 0.0,
        "last_sec": 125.0,
        "role": {
          "value": "cashier",
          "source": "auto"
        },
        "samples": []
      },
      {
        "track_id": "p2",
        "kind": "person",
        "object_class": null,
        "first_sec": 90.0,
        "last_sec": 96.0,
        "role": {
          "value": "unknown",
          "source": "auto"
        },
        "samples": []
      }
    ],
    "observations": [
      {
        "start_sec": 41.5,
        "end_sec": 47.5,
        "obs_id": "ob-1",
        "kind": "cash_in_hands",
        "subjects": [
          "p1",
          "o1"
        ],
        "zones": [
          "z-cash_desk-1"
        ],
        "score": 0.87,
        "frame_ids": [
          "fr-44200"
        ],
        "provenance": null,
        "model_id": null,
        "actor_id": null,
        "recipient_id": null
      }
    ],
    "coverage": [
      {
        "start_sec": 72.0,
        "end_sec": 80.0,
        "scope": "desk",
        "reason": "occluded"
      }
    ],
    "finding_set": null,
    "error": {
      "code": "VIDEO_DECODE_FAILED",
      "message": "Не удалось прочитать кадр.",
      "failed_phase": "perception",
      "at_sec": 48.0
    }
  }
};

export const ui_reevaluate_diff: ReevaluateResponse = {
  "finding_set": {
    "revision": 2,
    "basis": "reviewer",
    "created_at": "2026-10-17T07:30:00Z",
    "layout_revision": 1,
    "role_overrides": [],
    "verdict": "violation",
    "findings": [
      {
        "finding_id": "fd-288d1b1423",
        "rule_id": "V-COUNT-ABSENT",
        "type_label": "Пересчёт денег без клиента у кассы",
        "timecode_sec": 44.2,
        "window": {
          "start_sec": 41.5,
          "end_sec": 47.5
        },
        "basis": [
          "ob-1"
        ],
        "what_seen": "Кассир держит деньги, место клиента у окна видно и свободно.",
        "why_may_violate": "Условие требует проверки человеком.",
        "norm_refs": [
          "S231-p20"
        ],
        "description": "Кассир держит деньги, место клиента у окна видно и свободно. Синтетический пример структуры ответа.",
        "evidence": [
          {
            "frame_id": "fr-44200",
            "timestamp_sec": 44.2,
            "overlay_boxes": []
          }
        ],
        "strength": "moderate",
        "decision": null
      }
    ],
    "rules": [
      {
        "rule_id": "V-COUNT-ABSENT",
        "status": "partly_checkable",
        "not_checkable_windows": [
          {
            "start_sec": 72.0,
            "end_sec": 80.0,
            "reason": "Стол перекрыт."
          }
        ]
      },
      {
        "rule_id": "V-OUTSIDER",
        "status": "violation",
        "not_checkable_windows": []
      }
    ]
  },
  "diff": {
    "added": [],
    "removed": [
      "fd-3e26dc0700"
    ],
    "changed": [],
    "verdict_changed": false
  }
};

export const ui_rules: RulesResponse = {
  "rules": [
    {
      "rule_id": "V-COUNT-ABSENT",
      "type_label": "Пересчёт денег без клиента у кассы",
      "norm_refs": [
        "S231-p20"
      ],
      "requires_visible": [
        "cashier",
        "desk"
      ],
      "visible_sign": "Клиента у кассы не видно.",
      "why_may_violate": "Пример формата; не реальный результат анализа.",
      "not_claimed": "Сумма и документы.",
      "enabled": true,
      "sources": [
        {
          "clause_id": "S231-p20",
          "document": {
            "title": "Правила осуществления кассовых операций",
            "act": "Постановление № 231",
            "registry_code": "V1900019680",
            "url": "https://adilet.zan.kz/rus/docs/V1900019680",
            "edition_checked": "2026-10-17",
            "checked_at": "2026-10-17",
            "source_url": null,
            "source_version": null
          },
          "locator": "пункт 20",
          "quote": null,
          "verified_verbatim": false,
          "visual_test": "Присутствие клиента у кассы.",
          "not_visual": "Сумма операции."
        }
      ]
    }
  ],
  "not_checkable": []
};
