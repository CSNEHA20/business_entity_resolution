# Agent Instructions - Business Entity Resolution

## Repository & Remote Synchronization

All commits and code updates made to this repository must be pushed to **both** upstream repositories:

1. **Vishal Lakshmikanthan's Repository:**  
   `https://github.com/Vishallakshmikanthan/business-entity-resolution.git`

2. **CSNEHA20's Repository:**  
   `https://github.com/CSNEHA20/business_entity_resolution.git`

---

## Git Remote Configuration

The local repository is configured so that `origin` pushes to both repositories simultaneously.

### Remote URLs:
- **origin (fetch):** `https://github.com/Vishallakshmikanthan/business-entity-resolution.git`
- **origin (push):**
  - `https://github.com/Vishallakshmikanthan/business-entity-resolution.git`
  - `https://github.com/CSNEHA20/business_entity_resolution.git`
- **csneha (push/fetch):** `https://github.com/CSNEHA20/business_entity_resolution.git`

---

## Pushing Changes

Whenever changes are committed, run:
```bash
git push origin master
```
*(Or specify the active branch, e.g. `git push origin <branch_name>`)*

To ensure both remotes receive updates manually if needed:
```bash
git push https://github.com/Vishallakshmikanthan/business-entity-resolution.git master
git push https://github.com/CSNEHA20/business_entity_resolution.git master
```
