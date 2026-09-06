# Watcher Procedures

Procedure files tell the Vigil watcher what to look for in **coach mode**. The legacy desktop watcher provides observational hints via Piper TTS; model descriptions no longer advance steps. For Android checkpoints verified against fresh device UI evidence, see [the shared Beast adapter](../docs/MOBILE-BEAST.md).

---

## File Format

```json
{
  "name": "Human-readable procedure name",
  "description": "What this procedure covers",
  "version": "1.0",
  "steps": [
    {
      "id": 1,
      "description": "What the operator should do",
      "detect": "keyword the VLM should see on screen",
      "required": true,
      "hint": "Spoken hint if step is not progressing",
      "timeout_seconds": 60
    }
  ]
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Procedure name, spoken on load |
| `description` | no | Human description, not used at runtime |
| `version` | no | Version string for tracking |
| `steps` | yes | Ordered array of step objects |

### Step Fields

| Field | Required | Description |
|---|---|---|
| `id` | yes | Step number (integer, used in prompts) |
| `description` | yes | Spoken to operator as the current goal |
| `detect` | yes | Nonempty keyword used for observational coaching hints; not completion proof |
| `required` | no | If false, step can be skipped (default: true) |
| `hint` | no | Additional spoken guidance when step stalls |
| `timeout_seconds` | no | Not enforced yet — reserved for future timeout logic |

---

## How Detection Works

For each VLM call, the watcher sends this check to Cosmos-Reason2-8B:

```
MATCH: Does the screen show evidence of completing step N?
Answer YES or NO, then explain briefly.
```

Neither YES nor a matching keyword advances the step. Model prose cannot establish completion; the previous substring behavior is contained pending a native desktop postcondition adapter.

If the model says **NO**, the watcher speaks: *"Still on step N: [description]. Look for: [detect]."*

---

## Writing Good `detect` Keywords

The `detect` keyword provides context for coaching prompts. Choose keywords that are:

- **Specific** — `"New Object - User"` beats `"User"`
- **Screen-visible** — text that literally appears in a dialog title, button label, or UI element
- **Unambiguous** — avoid common words that appear in many contexts

For coaching context, prefer a specific UI label over a broad word:
- `"Active Directory"` matches `"Active Directory Users and Computers"`
- `"password"` matches any screen mentioning passwords

---

## Running Coach Mode

```bash
# Basic coaching
python watcher.py --mode coach --procedure watcher_procedures/it_basic.json

# With custom FPS and clip size
python watcher.py --mode coach --procedure watcher_procedures/it_basic.json --fps 2 --clip-frames 6

# Silent coaching (log only, no TTS)
python watcher.py --mode coach --procedure watcher_procedures/it_basic.json --no-tts --log /tmp/coach.log
```

---

## Included Procedures

| File | Domain | Steps | Description |
|---|---|---|---|
| `it_basic.json` | IT / SysAdmin | 8 | New user creation in Active Directory |
| `fire_watch.json` | Industrial Safety | 6 | Boiler startup sequence with SCADA monitoring |

---

## Building Domain Libraries

Planned procedure domains (see GitHub Issue #7):

| Domain | Examples |
|---|---|
| **IT / SysAdmin** | AD user setup, Group Policy, server reboots, patch cycles |
| **Medical** | Medication administration, patient intake, equipment sterilization |
| **Customer Service** | CRM ticket workflows, escalation paths, refund processing |
| **Physical / Field** | Equipment inspection checklists, safety walkaround |
| **Software QA** | Manual test case execution, deployment verification |
| **Finance** | Invoice approval workflows, reconciliation steps |

The procedure format is intentionally simple so domain experts (not developers) can write their own files.

---

*Part of the Vigil Vision-to-Action platform — rblake2320/vigil*
