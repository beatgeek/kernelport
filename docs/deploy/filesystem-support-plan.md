# Filesystem Support Plan (Lambda Cloud)

Intent: add optional per-deploy filesystem provisioning and cleanup to the Lambda Cloud deploy workflow so LuxTTS audio artifacts persist for the life of a deployment while controlling cost. The filesystem should be created in the same region selected for the instance, then attached at launch, and removed on teardown.

## Scope
- In: workflow updates to create/attach filesystem, new workflow inputs, cleanup path, and README guidance.
- Out: implementing any app-level storage logic or data migration.

## Action items
1. Review `.github/workflows/deploy-lambda.yml` to confirm launch payload shape and where filesystem fields should be added.
2. Validate Lambda Cloud filesystem API fields and constraints (name/size/region), plus attach semantics and instance launch payload schema.
3. Add workflow inputs for filesystem behavior (e.g., `filesystem_size_gb`, `filesystem_name_prefix`, `cleanup_filesystem`, and/or `filesystem_id` for override).
4. Add a filesystem create step using the selected region, capture the filesystem ID, and pass it in the instance launch request.
5. Add cleanup step(s) to delete filesystem if instance launch fails or if a separate teardown workflow is triggered.
6. Update `README.md` with filesystem usage, lifecycle, and any manual steps for recovery.
7. Define validation steps (API response checks; SSH into instance and verify mount/read/write).
8. Note edge cases: region mismatch, quota errors, cleanup retries, and dangling filesystem resources.

## Open questions
- Do we want a dedicated teardown workflow or cleanup in the same workflow after a failure only?
- What default filesystem size should we use for LuxTTS artifacts?
- Should we support overriding with an existing filesystem ID for special cases?
