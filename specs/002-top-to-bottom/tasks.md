# Tasks: Documentation and Code Hygiene Review

**Input**: Design documents from `/specs/002-top-to-bottom/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/documentation-structure.md, quickstart.md

**Tests**: This is a documentation feature - no automated tests required. Validation is manual via audit scripts and developer usability testing.

**Organization**: Tasks are grouped by user story to enable independent implementation and validation of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions
- Documentation files: `docs/`, `demos/*/`, `README.md`, `.github/`
- Scripts: `scripts/documentation_audit.sh`
- Configuration: `.gitignore`, `.markdownlint.json`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create audit tools and baseline documentation inventory

- [X] T001 Create documentation inventory script to list all .md files in docs_inventory.txt
- [X] T002 [P] Create scripts/documentation_audit.sh with executable permissions
- [X] T003 [P] Create TOC generation bash function in scripts/generate_toc.sh

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Audit current state before making changes

**⚠️ CRITICAL**: These audits must complete before ANY user story implementation can begin

- [X] T004 Run documentation inventory and count files by directory (top-level, docs/, demos/, specs/, .github/)
- [X] T005 Identify all files >100 lines that need table of contents
- [X] T006 [P] Detect duplicate content by comparing section headers across all .md files
- [X] T007 [P] Count current top-level directory items (target: reduce from 30 to <15)
- [X] T008 Audit current .gitignore coverage for Python/IDE/OS artifacts

**Checkpoint**: Baseline audit complete - user story implementation can now begin

---

## Phase 3: User Story 1 - Navigate Project Documentation Efficiently (Priority: P1) 🎯 MVP

**Goal**: Enable new developers to find setup instructions, API docs, and demo documentation within 2-10 minutes without confusion

**Independent Test**: Ask 3 new developers to:
1. Locate setup instructions (measure time, target <2 min)
2. Find API reference for ClassificationModel (measure time, target <2 min)
3. Navigate to credit_risk demo setup (measure time, target <2 min)
Success = 100% completion rate without asking for help

### Implementation for User Story 1

- [X] T009 [P] [US1] Add table of contents to README.md (currently >100 lines) with links to all major sections
- [X] T010 [P] [US1] Add table of contents to docs/architecture.md (>100 lines)
- [X] T011 [P] [US1] Add table of contents to docs/api_reference.md (>100 lines)
- [X] T012 [P] [US1] Add table of contents to docs/deployment.md (>100 lines)
- [X] T013 [P] [US1] Add table of contents to docs/user_guide.md (>100 lines)
- [X] T014 [US1] Update README.md to include Documentation Map section linking to docs/, demos/, specs/ with clear purpose statements
- [X] T015 [P] [US1] Ensure demos/credit_risk/README.md focuses on demo-specific setup (no project-wide content)
- [X] T016 [P] [US1] Ensure demos/fraud_detection/README.md focuses on demo-specific setup
- [X] T017 [P] [US1] Ensure demos/sales_forecasting/README.md focuses on demo-specific setup
- [X] T018 [P] [US1] Ensure demos/dna_similarity/README.md focuses on demo-specific setup
- [X] T019 [US1] Organize docs/ into subdirectories: docs/architecture/, docs/api/, docs/tutorials/ if not already structured
- [X] T020 [US1] Verify all TOC links work by testing in markdown preview

**Checkpoint**: User Story 1 complete - New developers can navigate documentation efficiently (verify with SC-001, SC-004 tests)

---

## Phase 4: User Story 2 - Maintain Clean Codebase Without Manual Cleanup (Priority: P2)

**Goal**: Prevent accidental commit of temporary files (__pycache__, .pytest_cache, IDE configs, OS files) via comprehensive .gitignore

**Independent Test**:
1. Run `pytest demos/credit_risk/tests/`
2. Open project in VS Code and PyCharm
3. Run `docker-compose up -d`
4. Run `git status`
Success = zero untracked files that should be ignored

### Implementation for User Story 2

- [X] T021 [US2] Audit current .gitignore and identify missing patterns for Python artifacts
- [X] T022 [US2] Add Python artifacts to .gitignore: __pycache__/, *.py[cod], .pytest_cache/, *.egg-info/
- [X] T023 [US2] Add IDE configurations to .gitignore: .vscode/, .idea/, *.swp, *.swo
- [X] T024 [US2] Add OS-specific files to .gitignore: .DS_Store, ._*, Thumbs.db
- [X] T025 [US2] Add IntegratedML-specific patterns to .gitignore: iris.log, iris.pid, iris.key, docker/volumes/iris_data/, docker/volumes/model_cache/
- [X] T026 [US2] Add model artifacts to .gitignore: *.pkl, *.joblib, checkpoints/
- [X] T027 [US2] Organize .gitignore with category comments (# Python artifacts, # IDE configurations, etc.)
- [X] T028 [US2] Validate .gitignore by running pytest and checking git status (SC-003 validation)
- [X] T029 [US2] Validate .gitignore covers ≥5 tools: pytest, VS Code, PyCharm, macOS, Docker, IRIS (SC-011 validation)

**Checkpoint**: User Story 2 complete - Code hygiene maintained automatically (verify with SC-003, SC-011 tests)

---

## Phase 5: User Story 3 - Understand Project Architecture from Documentation (Priority: P3)

**Goal**: Enable senior developers to understand base model hierarchy, design patterns, and IRIS integration from architecture docs without reading source code

**Independent Test**: Provide docs/architecture.md to senior developer unfamiliar with project. Ask:
1. "What's the base model hierarchy?"
2. "How does feature engineering work?"
3. "What's the model persistence strategy?"
Success = 100% correct answers without reading source code

### Implementation for User Story 3

- [X] T030 [P] [US3] Add base model hierarchy diagram to docs/architecture.md (ASCII or markdown format)
- [X] T031 [P] [US3] Add directory structure tree diagram to docs/architecture.md
- [X] T032 [P] [US3] Add data flow diagram (training → prediction) to docs/architecture.md
- [X] T033 [US3] Document feature engineering pipeline patterns in docs/architecture.md with code references to shared/models/
- [X] T034 [US3] Document model state management (_get_model_state, _set_model_state) in docs/architecture.md
- [X] T035 [US3] Document IRIS integration approach (SQL → JSON → Python) in docs/architecture.md
- [X] T036 [US3] Add ensemble architecture explanation (fraud detection demo pattern) to docs/architecture.md
- [X] T037 [US3] Ensure docs/architecture.md has ≥3 visual representations (SC-008 validation)

**Checkpoint**: User Story 3 complete - Architecture understandable from documentation (verify with SC-008 test)

---

## Phase 6: User Story 4 - Consolidate Redundant Documentation (Priority: P4)

**Goal**: Eliminate duplicate documentation by consolidating overlapping content and using cross-references instead of copy-pasting

**Independent Test**: Run documentation audit script to:
1. Compare section headers across all .md files
2. Verify each major topic has single authoritative source
Success = Zero duplicate sections found by `grep -rh "^### " docs/ demos/ | sort | uniq -d`

### Implementation for User Story 4

- [X] T039 [P] [US4] Remove duplicate API documentation from demos/credit_risk/README.md, replace with link to docs/api_reference.md
- [X] T040 [P] [US4] Remove duplicate API documentation from demos/fraud_detection/README.md, replace with link to docs/api_reference.md
- [X] T041 [P] [US4] Remove duplicate API documentation from demos/sales_forecasting/README.md, replace with link to docs/api_reference.md
- [X] T042 [P] [US4] Remove duplicate API documentation from demos/dna_similarity/README.md, replace with link to docs/api_reference.md
- [X] T043 [US4] Consolidate Docker setup instructions to single docs/DOCKER_SETUP.md (if multiple locations exist)
- [X] T044 [US4] Update all references to Docker setup to link to docs/DOCKER_SETUP.md
- [X] T045 [US4] Ensure docs/api_reference.md covers 100% of public methods in shared/models/ (SC-009 validation)
- [X] T046 [P] [US4] Verify demos/credit_risk/README.md is <500 lines (SC-007)
- [X] T047 [P] [US4] Verify demos/fraud_detection/README.md is <500 lines (SC-007)
- [X] T048 [P] [US4] Verify demos/sales_forecasting/README.md is <500 lines (SC-007)
- [X] T049 [P] [US4] Verify demos/dna_similarity/README.md is <500 lines (SC-007)
- [X] T050 [US4] Run documentation audit script to validate zero duplicates (SC-002)

**Checkpoint**: User Story 4 complete - Documentation consolidated (verify with SC-002, SC-007, SC-009 tests)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories and final validations

- [X] T051 [P] Create docs/CONTRIBUTING_DOCS.md with decision flowchart (where to add new documentation)
- [X] T052 [P] Reduce top-level directory to <15 items by organizing content into subdirectories (SC-006)
- [X] T053 Validate all internal links work using documentation audit script (SC-005)
- [X] T054 Run full documentation audit script (scripts/documentation_audit.sh) and verify all checks pass
- [ ] T055 [P] Create .markdownlint.json config with relaxed rules (MD013, MD033, MD041 disabled) - OPTIONAL
- [ ] T056 [P] Run markdownlint on all .md files if linter installed (SC-012 validation) - OPTIONAL
- [ ] T057 Conduct developer usability test with 3 new developers (SC-001, SC-010 validation)
- [ ] T058 Document final success criteria results in specs/002-top-to-bottom/VALIDATION_RESULTS.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories CAN proceed in parallel (different files, independent validations)
  - Or sequentially in priority order: US1 (P1) → US2 (P2) → US3 (P3) → US4 (P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - Independent from US1
- **User Story 3 (P3)**: Can start after Foundational - Independent from US1/US2
- **User Story 4 (P4)**: Can start after Foundational - May reference files from US1/US3 but independently testable

### Within Each User Story

- TOC tasks (T009-T013) can all run in parallel - different files
- Demo README tasks (T015-T018, T039-T042, T046-T049) can all run in parallel - different files
- Architecture diagram tasks (T030-T032) can run in parallel - different sections
- .gitignore category tasks (T022-T026) must run sequentially - same file

### Parallel Opportunities

- Phase 1 Setup: T002 and T003 can run in parallel
- Phase 2 Foundational: T006, T007, T008 can run in parallel
- Phase 3 US1: T009-T013 (TOC tasks) all parallel, T015-T018 (demo READMEs) all parallel
- Phase 5 US3: T030-T032 (architecture diagrams) all parallel
- Phase 6 US4: T039-T042 (remove duplicates from demos) all parallel, T046-T049 (verify lengths) all parallel
- Phase 7 Polish: T051, T052, T055, T056 can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all TOC tasks together (different files):
Task T009: "Add table of contents to README.md"
Task T010: "Add table of contents to docs/architecture.md"
Task T011: "Add table of contents to docs/api_reference.md"
Task T012: "Add table of contents to docs/deployment.md"
Task T013: "Add table of contents to docs/user_guide.md"

# Launch all demo README tasks together (different files):
Task T015: "Ensure demos/credit_risk/README.md focuses on demo-specific setup"
Task T016: "Ensure demos/fraud_detection/README.md focuses on demo-specific setup"
Task T017: "Ensure demos/sales_forecasting/README.md focuses on demo-specific setup"
Task T018: "Ensure demos/dna_similarity/README.md focuses on demo-specific setup"
```

---

## Parallel Example: User Story 4

```bash
# Launch all duplicate removal tasks together (different files):
Task T039: "Remove duplicate API docs from demos/credit_risk/README.md"
Task T040: "Remove duplicate API docs from demos/fraud_detection/README.md"
Task T041: "Remove duplicate API docs from demos/sales_forecasting/README.md"
Task T042: "Remove duplicate API docs from demos/dna_similarity/README.md"

# Launch all demo length validation tasks together (different files):
Task T046: "Verify demos/credit_risk/README.md is <500 lines"
Task T047: "Verify demos/fraud_detection/README.md is <500 lines"
Task T048: "Verify demos/sales_forecasting/README.md is <500 lines"
Task T049: "Verify demos/dna_similarity/README.md is <500 lines"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003) - ~10 minutes
2. Complete Phase 2: Foundational (T004-T008) - ~30 minutes (audit current state)
3. Complete Phase 3: User Story 1 (T009-T020) - ~45 minutes
4. **STOP and VALIDATE**: Test with new developer (SC-001: <10 min to find setup)
5. If validation passes, documentation navigation is READY

**Estimated MVP Time**: ~1.5 hours
**MVP Delivers**: Clear documentation structure, TOC in all major docs, demo-specific READMEs

### Incremental Delivery

1. **Setup + Foundational** → Baseline audit complete (~40 min)
2. **Add User Story 1** → Test independently → Navigation improved ✓
3. **Add User Story 2** → Test independently → Code hygiene automated ✓
4. **Add User Story 3** → Test independently → Architecture documented ✓
5. **Add User Story 4** → Test independently → Duplicates eliminated ✓
6. **Polish** → Final validation → Feature complete ✓

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (~40 min)
2. Once Foundational audit is done:
   - **Developer A**: User Story 1 (navigation) - ~45 min
   - **Developer B**: User Story 2 (.gitignore) - ~30 min
   - **Developer C**: User Story 3 (architecture) - ~40 min
   - **Developer D**: User Story 4 (duplicates) - ~50 min
3. Stories complete independently and can be tested in parallel
4. **Total parallel time**: ~90 min (vs. ~3 hours sequential)

---

## Success Criteria Validation Map

Each success criterion is validated by specific tasks:

- **SC-001**: New developer onboarding <10 min → T057 (usability test)
- **SC-002**: Zero duplicate documentation → T050 (audit script)
- **SC-003**: Zero untracked artifacts after development → T028 (.gitignore validation)
- **SC-004**: All files >100 lines have TOC → T009-T013, T020 (TOC additions + validation)
- **SC-005**: Zero broken links → T053 (link validation)
- **SC-006**: Top-level directory <15 items → T052 (directory reorganization)
- **SC-007**: Demo READMEs <500 lines → T046-T049 (length validation)
- **SC-008**: Architecture docs with ≥3 diagrams → T037 (diagram count validation)
- **SC-009**: API reference covers 100% public methods → T045 (API coverage validation)
- **SC-010**: 90%+ developer agreement on navigation → T057 (usability test)
- **SC-011**: .gitignore covers ≥5 tools → T029 (coverage validation)
- **SC-012**: Markdown linter zero errors → T056 (linter validation - OPTIONAL)

---

## Notes

- **[P] tasks** = different files, no dependencies - can run in parallel
- **[Story] label** maps task to specific user story (US1, US2, US3, US4) for traceability
- **No automated tests**: This is documentation work - validation is manual via audit scripts and developer testing
- **Each user story is independently completable and testable** - can ship US1 alone as MVP
- **Estimated total time**: ~2 hours sequential, ~90 minutes with 4 developers in parallel
- **Commit strategy**: Commit after each user story phase (4 commits) or after each major file change
- **Stop at any checkpoint** to validate story independently before proceeding
- **Avoid**: Editing same file in parallel (use sequential for .gitignore, README.md, architecture.md)
