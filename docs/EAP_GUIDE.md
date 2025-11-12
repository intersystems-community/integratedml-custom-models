# IntegratedML Custom Models - Early Access Program Guide

**Program Status**: Early Access Program (EAP)
**Target GA Release**: IRIS 2026.1
**EAP Launch**: January 2025
**Expected Duration**: Through 2026.1 release preparation

---

## Welcome to the Early Access Program

Thank you for participating in the **IntegratedML Custom Models Early Access Program**! You're among a select group of 5 participants who will help shape this groundbreaking capability before its public launch in IRIS 2026.1.

This guide explains what to expect during the EAP, how to provide feedback, and what will change between now and the general availability release.

---

## Table of Contents

- [What is IntegratedML Custom Models?](#what-is-integratedml-custom-models)
- [What is the Early Access Program?](#what-is-the-early-access-program)
- [EAP Timeline](#eap-timeline)
- [What's Included in EAP](#whats-included-in-eap)
- [What's Coming in GA (2026.1)](#whats-coming-in-ga-20261)
- [How to Get Started](#how-to-get-started)
- [How to Provide Feedback](#how-to-provide-feedback)
- [Support During EAP](#support-during-eap)
- [Expected Participant Activities](#expected-participant-activities)
- [Success Metrics](#success-metrics)
- [Frequently Asked Questions](#frequently-asked-questions)

---

## What is IntegratedML Custom Models?

**IntegratedML Custom Models** extends InterSystems IRIS IntegratedML with a powerful new capability: deploy your own Python machine learning models directly within SQL queries.

### Key Capabilities

- ✅ **Full Control**: Write custom preprocessing, feature engineering, and model training logic in Python
- ✅ **Any scikit-learn Compatible Model**: Use RandomForest, XGBoost, LightGBM, Prophet, or any model following the scikit-learn interface
- ✅ **In-Database Execution**: Models run directly in IRIS—no data movement, no export/import cycles
- ✅ **Same SQL Interface**: Use familiar `CREATE MODEL`, `TRAIN MODEL`, `VALIDATE MODEL`, and `PREDICT()` commands
- ✅ **Extends Existing IntegratedML**: Works alongside existing AutoML provider—choose the right tool for each use case

### How It Fits with IntegratedML

IntegratedML Custom Models is a **new provider** in the existing IntegratedML architecture:

```
IntegratedML (SQL Interface)
    ↓
ML Configuration
    ↓
Provider
    - %AutoML (automated, no code required)        ← Existing since 2020
    - DataRobot (third-party AutoML)               ← Existing
    - H2O (third-party AutoML)                     ← Existing
    - PMML (pre-trained model import)              ← Existing
    - Custom Models (Python code control)          ← NEW in IRIS 2025.2
```

**When to use AutoML vs Custom Models**:
- **Use AutoML** when you need quick models without ML expertise or custom code
- **Use Custom Models** when you need custom preprocessing, domain-specific logic, or third-party libraries

---

## What is the Early Access Program?

The **Early Access Program (EAP)** gives you exclusive access to IntegratedML Custom Models before the public launch. Your feedback will directly shape the final product.

### EAP Goals

1. **Validate real-world use cases** - Does Custom Models solve your ML deployment challenges?
2. **Identify documentation gaps** - Is the documentation clear and complete?
3. **Discover usability issues** - Can you install, configure, and use Custom Models without excessive support?
4. **Collect feature requests** - What additional capabilities would make this more valuable?
5. **Test production readiness** - Are security, performance, and operational considerations addressed?

### What You Get

- ✅ Early access to IntegratedML Custom Models capability
- ✅ Complete repository with 4 demo applications
- ✅ Direct feedback channel to the product team
- ✅ Opportunity to influence final product design
- ✅ Credit in GA release acknowledgments (if desired)

### What We Ask

- ⏰ **Time commitment**: ~5-10 hours over the EAP period
  - Initial installation and setup: 30 minutes - 1 hour
  - Explore demo applications: 2-4 hours
  - Evaluate for your use case: 2-4 hours
  - Provide structured feedback: 1-2 hours

- 📊 **Feedback commitment**:
  - Complete entry survey (5 minutes)
  - Report issues/bugs as you encounter them
  - Complete exit survey at end of EAP (15-20 minutes)
  - Optional: Participate in feedback calls or demos

- 🤝 **Confidentiality**: EAP content is pre-release; please don't share publicly until GA

---

## EAP Timeline

### Phase 1: Onboarding & Initial Exploration (Weeks 1-2)

**Week 1**:
- Receive EAP invitation and access to repository
- Complete installation on your system
- Run at least 2 demo applications
- Complete entry survey

**Week 2**:
- Explore remaining demo applications
- Review documentation for your intended use case
- Report initial feedback and issues

### Phase 2: Use Case Evaluation (Weeks 3-6)

- Evaluate Custom Models for your specific use case
- Attempt to create a custom model for your domain (optional but encouraged)
- Provide feedback on documentation gaps
- Test production deployment considerations

### Phase 3: Final Feedback & Wrap-Up (Week before 2026.1 GA)

- Complete exit survey
- Optional: Participate in feedback call with product team
- Review GA release notes and documentation
- Optional: Provide testimonial or use case for launch materials

**Note**: Exact timeline will be communicated via email. EAP duration approximately 6-8 weeks.

---

## What's Included in EAP

### ✅ Fully Functional Features

1. **Custom Model Creation**
   - Create custom Python model classes inheriting from base classes
   - Implement custom `fit()`, `predict()`, and `_validate_parameters()` methods
   - Use any scikit-learn compatible library

2. **SQL Integration**
   - `CREATE MODEL` with JSON USING clause for custom models
   - `TRAIN MODEL` with custom model parameters
   - `VALIDATE MODEL` for model evaluation
   - `PREDICT()` function for real-time predictions
   - `PROBABILITY()` function for classification confidence scores

3. **Model Types Supported**
   - Classification models (binary and multi-class)
   - Regression models
   - Ensemble models (combining multiple models)
   - Third-party models (Prophet, LightGBM, etc.)

4. **Demo Applications**
   - Credit Risk Assessment (classification with feature engineering)
   - Fraud Detection (ensemble with multiple sub-models)
   - Sales Forecasting (hybrid Prophet + LightGBM)
   - DNA Sequence Analysis (custom similarity algorithms)

5. **Documentation**
   - Quick start guide (5-minute setup)
   - User guide (step-by-step instructions)
   - API reference (complete base class documentation)
   - Architecture documentation (system design)
   - Deployment guide (production considerations)
   - 4 demo-specific tutorials

### ⚠️ Known Limitations in EAP

See [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) for complete list of current limitations and known bugs.

**Key Limitations**:
- Timeseries models not yet fully supported (workaround available via Prophet demo)
- Terminal/IRIS restart required after modifying custom model files
- Limited platform testing (macOS primary, Linux/Windows secondary)
- Documentation gaps in some advanced scenarios

### 🚧 Out of Scope for EAP

These features are **not included** in the EAP but may be considered for future releases:

- Automated model deployment pipelines
- Model versioning and rollback
- A/B testing framework for models
- Model monitoring dashboards (use existing IRIS monitoring)
- Automated hyperparameter tuning for custom models

---

## What's Coming in GA (2026.1)

See [`EAP_ROADMAP.md`](EAP_ROADMAP.md) for detailed roadmap from EAP to GA.

### Expected Changes Based on EAP Feedback

**Documentation Enhancements**:
- Address all documentation gaps reported during EAP
- Add production deployment case studies
- Expand troubleshooting guide based on EAP issues
- Create migration guide for AutoML users

**Stability Improvements**:
- Fix all critical bugs reported during EAP
- Improve error messages based on user feedback
- Enhance installation process for all platforms

**Feature Refinements**:
- Improve model lifecycle management based on usage patterns
- Enhance JSON USING clause validation
- Better integration with existing IntegratedML features

**Official Documentation**:
- Integration with docs.intersystems.com
- Addition to "Using IntegratedML" guide
- New "Custom Models Reference" guide
- SQL reference updates

---

## How to Get Started

### Prerequisites

Before you begin, ensure you have:

- ✅ **IRIS 2025.2 or later** (Community Edition or licensed)
- ✅ **Python 3.8+** installed
- ✅ **Docker** (recommended for quickest setup)
- ✅ **Git** for cloning the repository
- ✅ **5GB+ free disk space**

**Platform Support**:
- **Primary**: macOS (tested extensively)
- **Secondary**: Linux, Windows (tested but may have platform-specific issues)

### Quick Start (< 30 minutes)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/intersystems-community/integratedml-custom-models.git
   cd integratedml-custom-models
   ```

2. **Follow Installation Guide**

   See [`INSTALLATION.md`](INSTALLATION.md) for complete installation instructions.

   Quick setup (Docker):
   ```bash
   make setup  # Installs dependencies and starts IRIS
   make demo-credit  # Run first demo
   ```

3. **Verify Installation**

   Run the verification steps in the installation guide to confirm everything is working.

4. **Explore a Demo**

   Start with the Credit Risk demo:
   ```bash
   make demo-credit
   ```

   Follow along with the demo README: `demos/credit_risk/README.md`

5. **Review Documentation**

   - [`QUICK_GUIDE_CUSTOM_MODELS.md`](QUICK_GUIDE_CUSTOM_MODELS.md) - 5-minute overview
   - [`user_guide.md`](user_guide.md) - Step-by-step usage guide
   - [`architecture.md`](architecture.md) - How Custom Models works

### If You Get Stuck

1. Check [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for common issues
2. Review [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) for known limitations
3. Check [`EAP_FAQ.md`](EAP_FAQ.md) for frequently asked questions
4. Contact support (see "Support During EAP" section below)

---

## How to Provide Feedback

Your feedback is **critical** to making Custom Models successful. We've made it easy to provide feedback through multiple channels.

### Feedback Channels

#### 1. Structured Surveys (Primary - Recommended)

**Entry Survey** (5 minutes):
- Background and use case
- Installation experience
- Initial impressions

**Exit Survey** (15-20 minutes):
- Overall satisfaction
- Feature completeness
- Documentation quality
- Production readiness assessment
- Feature requests

**Survey Access**: Survey links will be provided by the Data Platforms Product Team via email.

#### 2. Email Feedback (Anytime)

**Email**: thomas.dyar@intersystems.com

Use email for:
- ✅ Questions during installation or usage
- ✅ Reporting bugs or issues
- ✅ Sharing use case requirements
- ✅ General feedback or suggestions

**Response Time**: Within 1-2 business days

#### 3. GitHub Issues (Optional - Under Exploration)

We're exploring using GitHub Issues for technical feedback. If enabled, use for:
- 🐛 Bug reports (technical issues)
- 📝 Documentation gaps or errors
- 💡 Feature requests

**Note**: GitHub Issues may not be enabled initially. Check repository for updates.

### What Makes Good Feedback?

#### Great Bug Reports Include:

```markdown
**Description**: What went wrong?
**Expected Behavior**: What should have happened?
**Actual Behavior**: What actually happened?
**Steps to Reproduce**: How can we reproduce this?
**Environment**: OS, Python version, IRIS version
**Error Messages**: Full error messages or logs
**Screenshots**: If applicable
```

#### Great Feature Requests Include:

```markdown
**Use Case**: What are you trying to accomplish?
**Current Limitation**: Why can't you do this today?
**Proposed Solution**: How would you like it to work?
**Alternatives Considered**: What workarounds have you tried?
**Impact**: How important is this to your use case?
```

#### Great Documentation Feedback Includes:

```markdown
**What Were You Trying to Do**: Your goal
**Where You Got Stuck**: Specific doc section
**What Was Unclear**: What confused you?
**Suggestion**: How could it be clearer?
```

---

## Support During EAP

### What Support Is Available?

**Email Support** (thomas.dyar@intersystems.com):
- Installation assistance
- Bug reports
- Usage questions
- Feature discussions

**Response Time**: 1-2 business days

**Documentation**:
- Comprehensive installation guide
- Troubleshooting guide for common issues
- EAP-specific FAQ
- Demo-specific tutorials

### What to Expect

✅ **We will**:
- Respond to emails within 1-2 business days
- Help troubleshoot installation issues
- Clarify documentation
- Acknowledge and track bug reports
- Consider feature requests for GA release

❌ **We cannot**:
- Provide 24/7 support (this is an EAP, not production)
- Write custom models for your specific use case (but we can provide guidance)
- Guarantee all feature requests will be implemented
- Support production deployments during EAP (evaluation only)

### Support Expectations

**Our Goal**: <1 support request for installation (i.e., 4-5 out of 5 participants install successfully without support).

If you need support, that's okay! But please:
1. Check troubleshooting documentation first
2. Check known issues documentation
3. Check FAQ
4. Then contact support with details

This helps us identify documentation gaps to fix before GA.

---

## Expected Participant Activities

### Minimum Commitment (5-6 hours)

1. **Installation & Setup** (30 min - 1 hour)
   - Install IRIS and dependencies
   - Clone repository
   - Run verification steps
   - Complete entry survey

2. **Explore Demos** (2-3 hours)
   - Run at least 2 of 4 demo applications
   - Review demo code and documentation
   - Understand SQL integration patterns

3. **Review Documentation** (1-2 hours)
   - Read Quick Start guide
   - Review User Guide sections relevant to your use case
   - Check API Reference for custom model development

4. **Provide Feedback** (1-2 hours)
   - Report any issues encountered
   - Complete exit survey
   - Share use case evaluation

### Recommended Activities (Additional 4-5 hours)

5. **Evaluate for Your Use Case** (2-3 hours)
   - Identify a real ML use case in your organization
   - Assess whether Custom Models could solve it
   - Document requirements or limitations

6. **Create a Custom Model** (2-3 hours)
   - Attempt to create a simple custom model for your domain
   - Follow one of the demo patterns
   - Document any issues or confusion

7. **Test Production Scenarios** (1-2 hours)
   - Review security best practices
   - Test performance with realistic data volumes
   - Evaluate operational considerations

### Optional Activities

8. **Feedback Call** (30 min - 1 hour)
   - Discuss your experience with product team
   - Demo your custom model (if created)
   - Provide verbal feedback on roadmap priorities

---

## Success Metrics

We're measuring the success of the EAP using these criteria:

### Installation Success

**Target**: 4-5 out of 5 participants install successfully in <30 minutes

**Measurement**: Entry survey + support tickets

### Documentation Quality

**Target**: Participants can complete demos and evaluate use cases using documentation only (minimal support needed)

**Measurement**: Support ticket volume + exit survey

### Feature Completeness

**Target**: Participants agree Custom Models solves real ML deployment challenges

**Measurement**: Exit survey + use case evaluation feedback

### Production Readiness

**Target**: Participants can assess production deployment feasibility

**Measurement**: Exit survey + operational feedback

### Overall Satisfaction

**Target**: >4.0/5.0 average satisfaction rating

**Measurement**: Exit survey

---

## Frequently Asked Questions

**For more detailed FAQs**, see [`EAP_FAQ.md`](EAP_FAQ.md).

### General Questions

**Q: Can I use Custom Models in production during the EAP?**
A: The EAP is for **evaluation purposes only**. We do not recommend production deployment until the GA release (IRIS 2026.1). However, you can evaluate production readiness during the EAP.

**Q: Will my EAP feedback be credited?**
A: Yes! If desired, we'll acknowledge EAP participants in GA release materials (with your permission).

**Q: Can I share about Custom Models publicly during EAP?**
A: Please keep EAP content confidential until the GA announcement. After GA, we encourage you to share your experiences!

### Technical Questions

**Q: Do I need to choose between AutoML and Custom Models?**
A: No! They work together. Use AutoML for quick models without code, and Custom Models when you need custom logic. You can use both in the same IRIS instance.

**Q: Can I use my existing Python ML code?**
A: Almost! As long as your model follows the scikit-learn interface (fit/predict methods), you can wrap it in a Custom Models class. See the demo applications for examples.

**Q: What if I find a critical bug?**
A: Please report it immediately via email (thomas.dyar@intersystems.com). We'll prioritize critical bugs for GA.

### Support Questions

**Q: What if I can't get it installed?**
A: Email thomas.dyar@intersystems.com with your error messages and environment details. We'll help troubleshoot.

**Q: How quickly will you respond to feedback?**
A: We aim for 1-2 business days for email responses. Surveys will be reviewed and acknowledged within 1 week.

---

## Next Steps

### Ready to Get Started?

1. **Read the Installation Guide**: [`INSTALLATION.md`](INSTALLATION.md)
2. **Review Known Issues**: [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md)
3. **Check the FAQ**: [`EAP_FAQ.md`](EAP_FAQ.md)
4. **Clone the Repository**: Get the code and start exploring
5. **Complete Entry Survey**: Let us know your background and use case

### Questions or Need Help?

- 📧 **Email**: thomas.dyar@intersystems.com
- 📚 **Documentation**: See `docs/` directory for all guides
- ❓ **FAQ**: [`EAP_FAQ.md`](EAP_FAQ.md)

---

## Thank You

Thank you for being part of the IntegratedML Custom Models Early Access Program. Your participation and feedback will directly shape the future of machine learning in InterSystems IRIS.

We're excited to see what you build!

**— The InterSystems Data Platforms Product Team**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-12
**Program Status**: Early Access Program (EAP)
