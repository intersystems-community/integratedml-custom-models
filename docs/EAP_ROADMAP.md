# IntegratedML Custom Models - EAP to GA Roadmap

**Program Status**: Early Access Program (EAP)
**Target GA Release**: IRIS 2026.1
**Last Updated**: 2025-01-12

---

## Purpose

This roadmap shows how IntegratedML Custom Models will evolve from the Early Access Program through the general availability (GA) release in IRIS 2026.1.

**Your feedback during EAP will directly influence this roadmap!**

---

## Table of Contents

- [Timeline Overview](#timeline-overview)
- [What's in EAP (Now)](#whats-in-eap-now)
- [What's Coming in GA (2026.1)](#whats-coming-in-ga-20261)
- [Post-GA Future Considerations](#post-ga-future-considerations)
- [How EAP Feedback Influences the Roadmap](#how-eap-feedback-influences-the-roadmap)
- [Feature Status Tracking](#feature-status-tracking)

---

## Timeline Overview

```
January 2025          February-March 2025         Q2 2026              Q3 2026+
    ↓                         ↓                      ↓                    ↓
┌─────────┐          ┌──────────────────┐     ┌──────────┐       ┌──────────────┐
│   EAP   │  ─────→  │  EAP Iteration   │ ──→ │ 2026.1   │  ───→ │  Post-GA     │
│ Launch  │          │  & Refinement    │     │   GA     │       │  Enhancements│
└─────────┘          └──────────────────┘     └──────────┘       └──────────────┘
     │                        │                     │                    │
     │                        │                     │                    │
 • 5 EAP                  • Bug fixes          • Public              • Advanced
   participants           • Doc updates          release              features
 • Core features          • Feature            • Full docs          • Based on
 • 4 demos                  refinement         • Production           usage data
 • Documentation          • Performance          ready              • Community
                            tuning             • Stable API           feedback
```

**Key Milestones**:
- ✅ **EAP Launch**: January 2025 (5 participants)
- 🔄 **EAP Iteration Period**: February-March 2025 (6-8 weeks)
- 🎯 **Feature Freeze**: ~2 months before 2026.1 GA
- 🚀 **GA Release**: Q2 2026 (IRIS 2026.1)
- 📈 **Post-GA Enhancements**: Q3 2026 and beyond

---

## What's in EAP (Now)

### Core Functionality ✅

**Custom Model Integration**:
- ✅ Create custom Python model classes
- ✅ Inherit from `ClassificationModel`, `RegressionModel`, `EnsembleModel` base classes
- ✅ Implement custom `fit()`, `predict()`, `_validate_parameters()` methods
- ✅ Use any scikit-learn compatible library

**SQL Integration**:
- ✅ `CREATE MODEL` with JSON USING clause
- ✅ `TRAIN MODEL` with custom parameters
- ✅ `VALIDATE MODEL` for evaluation
- ✅ `PREDICT()` function for predictions
- ✅ `PROBABILITY()` function for classification confidence

**Model Types**:
- ✅ Classification (binary and multi-class)
- ✅ Regression
- ✅ Ensemble (multiple model voting/averaging)
- ✅ Third-party library integration (Prophet, LightGBM, XGBoost)

**Demo Applications** (4 complete examples):
- ✅ Credit Risk Assessment
- ✅ Fraud Detection (Ensemble)
- ✅ Sales Forecasting (Hybrid Prophet + LightGBM)
- ✅ DNA Sequence Similarity

**Documentation**:
- ✅ Quick start guide
- ✅ User guide
- ✅ API reference
- ✅ Architecture documentation
- ✅ Deployment guide
- ✅ Demo tutorials
- ✅ EAP-specific guides (GUIDE, KNOWN_ISSUES, ROADMAP, FAQ)

### Known Limitations ⚠️

See [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) for complete list.

**Key Limitations**:
- ⚠️ Timeseries models require wrapper pattern
- ⚠️ Terminal restart needed after model changes
- ⚠️ Primary macOS support, secondary Linux/Windows
- ⚠️ Some error messages need improvement
- ⚠️ Production documentation incomplete

---

## What's Coming in GA (2026.1)

### Stability & Quality Improvements 🎯

#### 1. Bug Fixes & Error Handling

**Based on EAP Feedback**:
- 🔧 Fix all critical bugs reported during EAP
- 🔧 Enhanced error messages for missing model methods
- 🔧 Better JSON USING clause validation
- 🔧 Improved serialization error messages
- 🔧 Clear error messages for configuration issues

**Status**: Prioritized based on severity and user impact

**Timeline**: Throughout EAP period and feature freeze

---

#### 2. Installation & Setup Improvements

**Automation**:
- 🔧 Automated symlink creation during installation
- 🔧 Installation verification script
- 🔧 Platform-specific installation guides (macOS, Linux, Windows)
- 🔧 Docker compose improvements for volume permissions

**Cross-Platform Support**:
- 🔧 Full testing on macOS, Linux (Ubuntu 22.04+), Windows (WSL2)
- 🔧 Platform-specific troubleshooting
- 🔧 Consistent behavior across platforms

**Status**: Based on EAP installation feedback

**Timeline**: GA release package

---

#### 3. Performance Optimizations

**Query Performance**:
- 🔧 Optimize `PREDICT()` function for large result sets
- 🔧 Batch prediction performance improvements
- 🔧 Model loading time optimization

**Training Performance**:
- 🔧 Investigate async training for long-running models
- 🔧 Progress indicators for training (if feasible)
- 🔧 Memory usage optimization

**Status**: Benchmarking during EAP, optimizations for GA

**Timeline**: Continuous improvement through EAP

**Target Metrics**:
- `PREDICT()` latency: <50ms per row (current: varies by model)
- Model load time: <5 seconds for typical models (current: varies)
- Training throughput: Support datasets up to 1M rows without timeout

---

### Documentation Enhancements 📚

#### 4. Expanded Documentation

**Production Readiness**:
- 📝 Complete security best practices guide (expanded)
- 📝 Complete performance tuning guide (expanded)
- 📝 Operational runbook templates
- 📝 Monitoring and alerting setup guide

**Migration & Decision Guides**:
- 📝 AutoML to Custom Models migration guide
- 📝 Decision flowchart: When to use AutoML vs Custom Models
- 📝 Side-by-side comparison examples

**Advanced Topics**:
- 📝 Model state management best practices
- 📝 Complex ensemble patterns
- 📝 Custom metrics and loss functions
- 📝 Model interpretability (SHAP, LIME integration)
- 📝 A/B testing patterns

**Troubleshooting**:
- 📝 Expanded troubleshooting guide based on EAP issues
- 📝 Platform-specific troubleshooting sections
- 📝 Diagnostic information collection guide

**Status**: Actively gathering gaps during EAP

**Timeline**: Final documentation complete at GA

---

#### 5. Official InterSystems Documentation Integration

**docs.intersystems.com**:
- 📝 Addition to "Using IntegratedML" guide
- 📝 New "Custom Models Reference" guide
- 📝 SQL syntax reference updates
- 📝 Integration with existing AutoML documentation
- 📝 Version-specific documentation (2026.1+)

**Content Delivery**:
- 📝 PDF downloads for offline use
- 📝 Searchable documentation
- 📝 Cross-linked with related IRIS features

**Status**: Coordination with documentation team during EAP

**Timeline**: GA release documentation

---

### Feature Enhancements 🚀

#### 6. Developer Experience Improvements

**Model Development**:
- 🚀 Improved base class interfaces
- 🚀 More helper methods in base classes
- 🚀 Better parameter validation
- 🚀 Enhanced debugging capabilities

**Testing & Validation**:
- 🚀 Model testing utilities
- 🚀 Validation helpers
- 🚀 Example unit tests for each model type

**Status**: Based on EAP developer feedback

**Timeline**: GA release

---

#### 7. Enhanced Examples & Templates

**New Demo Applications** (if time permits):
- 🚀 Healthcare example (patient risk scoring)
- 🚀 Manufacturing example (predictive maintenance)
- 🚀 Additional industry-specific examples

**Model Templates**:
- 🚀 Template for custom preprocessing
- 🚀 Template for ensemble models
- 🚀 Template for third-party library integration

**Status**: Based on EAP use case feedback

**Timeline**: GA release or post-GA

---

### API Stability 🔒

#### 8. API Freeze & Backward Compatibility Commitment

**GA Release Commitment**:
- 🔒 API freeze at GA - no breaking changes after 2026.1
- 🔒 Backward compatibility guarantee for base classes
- 🔒 Deprecation policy for any future changes (minimum 2 major versions notice)
- 🔒 Semantic versioning for Python package

**Current API Status**:
- ⚠️ EAP: API is subject to change based on feedback
- ✅ GA: API is stable and supported

**Status**: API will be finalized based on EAP feedback

**Timeline**: Locked at feature freeze (before GA)

---

## Post-GA Future Considerations

These features are **under consideration** for post-GA releases (2026.2+), based on EAP and GA user feedback.

### Advanced Features (Under Consideration) 💡

#### 1. Timeseries Native Support

**Description**: Native support for timeseries models without wrapper patterns

**Use Cases**:
- ARIMA, SARIMA models
- Prophet as standalone model
- Facebook Neural Prophet
- Time-series specific preprocessing

**Status**: Investigating feasibility based on EAP sales forecasting demo feedback

**Timeline**: Post-GA if high demand

---

#### 2. Model Versioning & Rollback

**Description**: Track model versions and roll back to previous versions

**Features**:
- Model version history
- Rollback to previous model version
- A/B testing between model versions
- Version-specific PREDICT()

**Status**: Based on production deployment feedback

**Timeline**: Post-GA (2026.2 or later)

---

#### 3. Hot Reload for Model Changes

**Description**: Update models without restarting IRIS terminal

**Benefits**:
- Faster iterative development
- No downtime for model updates
- Better developer experience

**Challenge**: Architecture constraint requires investigation

**Status**: Under investigation

**Timeline**: Post-GA if feasible

---

#### 4. Model Monitoring & Drift Detection

**Description**: Built-in monitoring for model performance and data drift

**Features**:
- Prediction distribution monitoring
- Data drift detection
- Model performance metrics over time
- Alerting for performance degradation

**Status**: Requires integration with IRIS monitoring

**Timeline**: Post-GA (2026.2+)

---

#### 5. Model Marketplace / Template Library

**Description**: Pre-built model templates for common use cases

**Features**:
- Industry-specific model templates
- Community-contributed models
- Best practice examples
- Copy-paste ready implementations

**Status**: Depends on community engagement post-GA

**Timeline**: Post-GA (community-driven)

---

#### 6. AutoML Integration for Custom Models

**Description**: Use AutoML features (feature engineering, model selection) with custom models

**Features**:
- Automatic feature engineering for custom models
- Hyperparameter tuning for custom models
- Model selection across custom and AutoML models

**Status**: Architectural exploration needed

**Timeline**: Post-GA (if feasible)

---

#### 7. Streaming / Incremental Learning Support

**Description**: Update models with new data without full retraining

**Use Cases**:
- Online learning
- Incremental model updates
- Real-time model adaptation

**Challenge**: Requires models to support incremental learning

**Status**: Based on production use case feedback

**Timeline**: Post-GA (2026.2+)

---

## How EAP Feedback Influences the Roadmap

Your feedback during the EAP will help us prioritize features and improvements for GA and beyond.

### How We Use Your Feedback

**Bug Reports** → Prioritized for GA bug fixes
- Critical bugs: Fixed before GA
- High priority bugs: Fixed before GA
- Medium priority bugs: Fixed for GA or early patch
- Low priority bugs: Tracked for future releases

**Feature Requests** → Evaluated for roadmap inclusion
- High impact, high feasibility: Considered for GA
- High impact, medium feasibility: Post-GA roadmap
- Nice-to-have: Community contribution or future releases

**Documentation Gaps** → Filled before GA
- Critical gaps (block usage): Fixed immediately
- Important gaps: Fixed before GA
- Nice-to-have: Filled before or shortly after GA

**Use Case Feedback** → Shapes future direction
- Common patterns: Become templates/examples
- Industry-specific needs: Inform demo priorities
- Production requirements: Inform operational features

### Feedback Channels

See [`EAP_GUIDE.md#how-to-provide-feedback`](EAP_GUIDE.md#how-to-provide-feedback) for details.

**Survey** (Structured feedback):
- Entry survey: Your background and use cases
- Exit survey: Overall satisfaction and priorities

**Email** (Anytime):
- thomas.dyar@intersystems.com

**GitHub Issues** (If enabled):
- Bug reports
- Feature requests
- Documentation improvements

---

## Feature Status Tracking

### GA Release (2026.1) - Committed Features

| Feature | Status | Priority | Timeline |
|---------|--------|----------|----------|
| **Stability & Quality** |
| Fix critical bugs from EAP | 🔄 In Progress | P0 | Before GA |
| Enhanced error messages | 🔄 In Progress | P1 | GA |
| Platform testing (Mac/Linux/Win) | 🔄 In Progress | P1 | GA |
| **Installation** |
| Automated installation scripts | 📋 Planned | P1 | GA |
| Platform-specific guides | 📋 Planned | P1 | GA |
| Docker improvements | 📋 Planned | P2 | GA |
| **Performance** |
| PREDICT() optimization | 📋 Planned | P1 | GA |
| Model loading optimization | 📋 Planned | P2 | GA |
| Training performance improvements | 📋 Planned | P2 | GA |
| **Documentation** |
| Production deployment guide | 📋 Planned | P1 | GA |
| Migration guide (AutoML → Custom) | 📋 Planned | P1 | GA |
| Security best practices | 📋 Planned | P1 | GA |
| Performance tuning guide | 📋 Planned | P1 | GA |
| docs.intersystems.com integration | 📋 Planned | P1 | GA |
| Advanced examples | 📋 Planned | P2 | GA |
| **Developer Experience** |
| Improved base classes | 📋 Planned | P2 | GA |
| Testing utilities | 📋 Planned | P2 | GA |
| Model templates | 📋 Planned | P2 | GA |
| **API** |
| API freeze | 📋 Planned | P0 | Feature Freeze |
| Backward compatibility guarantee | 📋 Planned | P0 | GA |

**Legend**:
- ✅ Complete
- 🔄 In Progress (active development)
- 📋 Planned (scheduled for GA)
- 💡 Under Consideration (post-GA)
- ❓ Investigating (feasibility study)

---

### Post-GA (2026.2+) - Under Consideration

| Feature | Interest Level | Complexity | Potential Timeline |
|---------|---------------|------------|-------------------|
| Timeseries native support | High | Medium | 2026.2 |
| Model versioning | Medium | Medium | 2026.2-2026.3 |
| Hot reload | High | High | TBD |
| Model monitoring | Medium | High | 2026.3+ |
| Model marketplace | Low | Low | Community |
| AutoML integration | Low | High | TBD |
| Incremental learning | Medium | High | 2026.3+ |

**Notes**:
- Interest level based on initial stakeholder input
- Will be updated based on EAP feedback
- Timeline estimates are preliminary

---

## Roadmap Principles

### Our Commitments

**For GA (2026.1)**:
1. ✅ Fix all critical bugs from EAP
2. ✅ Complete documentation gaps
3. ✅ Ensure cross-platform compatibility
4. ✅ Provide production-ready guidance
5. ✅ Stable, backward-compatible API

**For Post-GA**:
6. ✅ Listen to community feedback
7. ✅ Prioritize based on real-world usage
8. ✅ Maintain backward compatibility
9. ✅ Regular updates and improvements
10. ✅ Open to community contributions

### What Won't Change

**Core Value Propositions** (permanent):
- ✅ SQL-first interface (same IntegratedML commands)
- ✅ In-database execution (no data movement)
- ✅ scikit-learn compatibility (standard interface)
- ✅ Full Python control (custom preprocessing, models)
- ✅ Works alongside AutoML (choose right tool for job)

---

## Questions or Suggestions?

**Have ideas for the roadmap?**
- 📧 Email: thomas.dyar@intersystems.com
- 📊 Survey: Share priorities in exit survey
- 💡 Feature requests: Describe your use case

**Want to influence priorities?**
- Complete EAP surveys with detailed feedback
- Share your use cases and requirements
- Participate in optional feedback calls
- Report what's working well (not just bugs!)

---

## Thank You

Your participation in the EAP directly shapes the future of IntegratedML Custom Models. Every piece of feedback helps us build a better product for the entire IRIS community.

**We're excited to see what you build and where this feature goes!**

**— The InterSystems Data Platforms Product Team**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-12
**Next Update**: Based on EAP progress and feedback

**Latest Version**: https://github.com/intersystems-community/integratedml-custom-models
