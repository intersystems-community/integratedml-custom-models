# IntegratedML Custom Models - EAP Frequently Asked Questions

**Program Status**: Early Access Program (EAP)
**Target GA Release**: IRIS 2026.1
**Last Updated**: 2025-01-12

---

## Purpose

This document answers frequently asked questions about participating in the IntegratedML Custom Models Early Access Program.

For technical questions about using Custom Models, see the main documentation:
- [`user_guide.md`](user_guide.md) - How to use Custom Models
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - Common technical issues
- [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) - Known limitations and bugs

---

## Table of Contents

- [Program Overview](#program-overview)
- [Getting Started](#getting-started)
- [Technical Questions](#technical-questions)
- [Feedback & Support](#feedback--support)
- [Timeline & Roadmap](#timeline--roadmap)
- [Production Use](#production-use)
- [After EAP](#after-eap)

---

## Program Overview

### What is the EAP?

**Q: What is an Early Access Program?**

A: An Early Access Program (EAP) gives select users exclusive access to new features before public launch. You get to try Custom Models early, provide feedback, and help shape the final product.

**Benefits**:
- ✅ Early access to cutting-edge ML capabilities
- ✅ Direct influence on product design
- ✅ Direct communication with product team
- ✅ Credit in GA release (if desired)

**Responsibilities**:
- ⏰ ~5-10 hours time commitment
- 📊 Provide structured feedback
- 🤝 Respect confidentiality until GA

---

**Q: Why was I selected for the EAP?**

A: You were selected because you have relevant ML/data science experience with IRIS or expressed interest in advanced IntegratedML capabilities. We're looking for participants who can:

- Evaluate Custom Models for real-world use cases
- Provide technical feedback on usability
- Test documentation clarity
- Identify gaps or missing features

---

**Q: How many participants are in the EAP?**

A: 5 participants in this initial EAP cohort. This small group allows for:

- Personalized support and attention
- Direct communication with product team
- Manageable feedback volume for rapid iteration
- Higher quality feedback per participant

---

### What is IntegratedML Custom Models?

**Q: What problem does Custom Models solve?**

A: Custom Models solves the "last mile" problem for advanced ML use cases:

**Problem**: IntegratedML AutoML is great for quick models without code, but what if you need:
- Custom preprocessing or feature engineering?
- Domain-specific algorithms?
- Third-party libraries (Prophet, LightGBM, XGBoost)?
- Full control over model training?

**Solution**: Custom Models lets you write custom Python models while keeping IntegratedML's core benefits:
- ✅ Same SQL interface (`CREATE MODEL`, `PREDICT()`)
- ✅ In-database execution (no data movement)
- ✅ Real-time predictions on live data
- ✅ Works alongside AutoML (use the right tool for each job)

---

**Q: How does Custom Models relate to AutoML?**

A: They're **complementary**, not competitive:

**Use AutoML when**:
- You need quick models without writing code
- You don't have ML expertise
- Standard algorithms work for your use case
- You want automatic feature engineering

**Use Custom Models when**:
- You need custom preprocessing logic
- You want to use specific third-party libraries
- You have domain-specific algorithms
- You need full control over model training

**Both work together** in the same IRIS instance. Choose the right tool for each use case.

---

**Q: Is Custom Models replacing AutoML?**

A: **No!** Custom Models is an **addition** to IntegratedML, not a replacement for AutoML.

- AutoML remains the recommended choice for quick models without code
- Custom Models extends IntegratedML for advanced use cases
- You can use both in the same IRIS instance
- Same SQL commands work with both providers

---

## Getting Started

### Installation

**Q: What do I need to install?**

A: **Prerequisites**:
- IRIS 2025.2 or later
- Python 3.8+ (3.11+ recommended)
- Docker (recommended) or local IRIS installation
- Git
- 5GB+ free disk space

**See** [`INSTALLATION.md`](INSTALLATION.md) for complete installation guide.

---

**Q: How long does installation take?**

A: **Target**: <30 minutes for most users

**Typical timeline**:
- Docker setup: 15-20 minutes (recommended)
- Local installation: 20-30 minutes
- Running first demo: 5-10 minutes

If installation takes longer than 30 minutes, please contact support - we want to understand why!

---

**Q: Which platform should I use?**

A: **Recommended priority**:

1. **macOS** (primary support, most tested)
2. **Linux** (Ubuntu 22.04+, secondary support)
3. **Windows** (WSL2 or Docker, limited testing)

**Best experience**: Use Docker on any platform for most consistent results.

---

**Q: Do I need a licensed IRIS installation?**

A: **No** - IRIS Community Edition works fine for the EAP.

However, if you have a licensed installation and want to evaluate Custom Models in your production environment, that's encouraged (for evaluation only, not production deployment during EAP).

---

### First Steps

**Q: Where should I start?**

A: **Recommended path**:

1. **Read [`EAP_GUIDE.md`](EAP_GUIDE.md)** - Understand the program (15 min)
2. **Follow [`INSTALLATION.md`](INSTALLATION.md)** - Get set up (20-30 min)
3. **Run Credit Risk demo** - See it work (15 min)
   ```bash
   make demo-credit
   ```
4. **Review [`QUICK_GUIDE_CUSTOM_MODELS.md`](QUICK_GUIDE_CUSTOM_MODELS.md)** - 5-minute overview
5. **Complete entry survey** - Tell us about your background (5 min)

**Total time to first success**: ~1 hour

---

**Q: Which demo should I try first?**

A: **Start with Credit Risk** - it's the simplest and shows core concepts:

**Demo order (easiest to most complex)**:
1. **Credit Risk** (classification, feature engineering)
2. **DNA Similarity** (custom algorithms, pattern matching)
3. **Fraud Detection** (ensemble, multiple sub-models)
4. **Sales Forecasting** (hybrid Prophet + LightGBM, most complex)

Run demos with: `make demo-credit`, `make demo-fraud`, etc.

---

### Evaluation

**Q: What should I evaluate during the EAP?**

A: **Three levels of evaluation**:

**Level 1 - Minimum** (~2 hours):
- Run 2-3 demos successfully
- Review documentation for clarity
- Report any installation/usage issues

**Level 2 - Recommended** (~5-6 hours):
- All of Level 1, plus:
- Identify a real use case in your organization
- Assess whether Custom Models could solve it
- Review production deployment considerations

**Level 3 - Advanced** (~10+ hours):
- All of Level 2, plus:
- Create a simple custom model for your domain
- Test with realistic data volumes
- Evaluate operational requirements (monitoring, security, performance)

**Choose the level that fits your available time.**

---

**Q: Should I create a custom model for my use case?**

A: **Optional but encouraged!**

**If you have time** (extra 2-3 hours):
- Pick a simple ML use case from your domain
- Try creating a custom model following demo patterns
- Document what worked well and what was confusing

**If time is limited**:
- Just evaluate whether Custom Models could work for your use cases
- Provide feedback on documentation and existing demos
- That's still very valuable feedback!

---

## Technical Questions

### Using Custom Models

**Q: Can I use any Python ML library?**

A: **Almost!** As long as the library follows the scikit-learn interface (has `fit()` and `predict()` methods), you can wrap it.

**Confirmed working**:
- ✅ scikit-learn
- ✅ LightGBM
- ✅ XGBoost
- ✅ Prophet
- ✅ CatBoost
- ✅ Any sklearn-compatible library

**Not directly supported**:
- ❌ TensorFlow/Keras (not sklearn-compatible, but can be wrapped)
- ❌ PyTorch (not sklearn-compatible, but can be wrapped)
- ⚠️ Pure timeseries libraries (need wrapper pattern, see sales demo)

**See demos for integration examples.**

---

**Q: How do I debug my custom model?**

A: **Recommended workflow**:

1. **Develop outside IRIS first**:
   ```python
   # Test your model with pytest
   pytest demos/my_demo/tests/
   ```

2. **Deploy to IRIS**:
   ```bash
   # Copy model file to IRIS
   docker cp my_model.py iml-custom-models-iris:/path/to/models/
   ```

3. **Test in SQL**:
   ```sql
   CREATE MODEL MyModel PREDICTING (target) FROM MyTable
   USING '{"model_name": "MyModel", ...}'

   TRAIN MODEL MyModel
   ```

4. **Check IRIS logs**:
   ```bash
   docker logs iml-custom-models-iris
   ```

**See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for common debugging scenarios.**

---

**Q: Why do I need to restart IRIS after changing my model?**

A: This is a **current limitation** of the model loading architecture:

**Why**: Python modules are loaded once and cached. Changes to `.py` files aren't picked up until IRIS restarts.

**Workaround**:
- Develop and test models outside IRIS first
- Only deploy to IRIS when model is working
- Restart IRIS terminal or container after deploying:
  ```bash
  docker restart iml-custom-models-iris
  ```

**Future**: Hot-reload is under investigation for post-GA release.

**See**: [`EAP_KNOWN_ISSUES.md#bug-006`](EAP_KNOWN_ISSUES.md#development-workflow-limitations)

---

**Q: Can I use my existing Python ML code?**

A: **Mostly yes**, with some adaptation:

**If your code uses sklearn interface**:
```python
# Your existing code
class MyModel:
    def fit(self, X, y): ...
    def predict(self, X): ...
```

**Wrap it for IntegratedML**:
```python
from shared.models.classification import ClassificationModel

class MyCustomModel(ClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_model = MyExistingModel()

    def fit(self, X, y):
        X_processed = self._engineer_features(X)
        return self.my_model.fit(X_processed, y)

    def predict(self, X):
        X_processed = self._engineer_features(X)
        return self.my_model.predict(X_processed)
```

**See demo applications for complete patterns.**

---

### SQL Integration

**Q: What SQL commands work with Custom Models?**

A: **Same commands as AutoML**:

```sql
-- Create model definition
CREATE MODEL MyModel
PREDICTING (target_column)
FROM MyTable
USING '{"model_name": "MyCustomModel", ...}'

-- Train model
TRAIN MODEL MyModel

-- Validate model
VALIDATE MODEL MyModel FROM ValidationTable

-- Make predictions
SELECT id,
       PREDICT(MyModel) as prediction,
       PROBABILITY(MyModel FOR 1) as confidence
FROM NewData

-- Drop model
DROP MODEL MyModel
```

**See [`user_guide.md`](user_guide.md) for complete SQL reference.**

---

**Q: How do I pass parameters to my custom model?**

A: **Use the JSON USING clause**:

```sql
TRAIN MODEL MyModel
USING '{
    "model_name": "MyCustomModel",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_feature_1": 1,
        "threshold": 0.75,
        "custom_setting": "value"
    }
}'
```

**In your model**:
```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Access user_params
    params = kwargs.get('user_params', {})
    self.enable_feature_1 = params.get('enable_feature_1', 0)
    self.threshold = params.get('threshold', 0.5)
```

**See [`api_reference.md`](api_reference.md) for parameter handling details.**

---

## Feedback & Support

### Providing Feedback

**Q: How do I provide feedback?**

A: **Three channels**:

1. **Survey** (structured, recommended):
   - Entry survey: Background and initial impressions
   - Exit survey: Overall evaluation and priorities
   - Survey links provided by Data Platforms Product Team

2. **Email** (anytime):
   - thomas.dyar@intersystems.com
   - Use for questions, bug reports, suggestions

3. **GitHub Issues** (optional, if enabled):
   - Bug reports
   - Feature requests
   - Documentation gaps

**See [`EAP_GUIDE.md#how-to-provide-feedback`](EAP_GUIDE.md#how-to-provide-feedback) for details.**

---

**Q: What makes good feedback?**

A: **Specific and actionable**:

**Great feedback**:
- ✅ "Installation failed on Windows 11 with error: [specific error]. I was following step 3 of INSTALLATION.md."
- ✅ "I couldn't figure out how to pass parameters to my model. Documentation says 'use user_params' but doesn't show an example."
- ✅ "For my use case (customer churn prediction), I need to load pre-trained embeddings. Is this possible?"

**Less helpful feedback**:
- ❌ "Installation didn't work"
- ❌ "Documentation is confusing"
- ❌ "Needs more features"

**Include**:
- What you were trying to do
- What went wrong (specific error messages)
- Your environment (OS, Python version, etc.)
- Suggestions for improvement

---

**Q: Will my feedback be implemented?**

A: **It depends**:

**Critical bugs**: Fixed before GA (guaranteed)
**High-impact improvements**: Strong consideration for GA
**Feature requests**: Evaluated for GA or post-GA roadmap
**Nice-to-haves**: Tracked for future releases

**We review all feedback** and will acknowledge receipt within 1-2 business days.

Your feedback directly influences priorities. The more specific and detailed, the more likely we can act on it!

---

### Getting Support

**Q: What if I get stuck?**

A: **Support process**:

1. **Check documentation**:
   - [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - Common issues
   - [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) - Known limitations
   - [`EAP_FAQ.md`](EAP_FAQ.md) - This document

2. **Search repository**: Check if others encountered same issue

3. **Contact support**:
   - Email: thomas.dyar@intersystems.com
   - Response time: 1-2 business days

**Include in support request**:
- What you were trying to do
- Steps to reproduce
- Error messages (full text)
- Environment details (OS, Python version, IRIS version)
- Screenshots if applicable

---

**Q: What support response time should I expect?**

A: **Target response time**: 1-2 business days

**What we provide**:
- ✅ Help with installation issues
- ✅ Clarify documentation
- ✅ Troubleshoot bugs
- ✅ Guidance on model development

**What we don't provide**:
- ❌ 24/7 support (EAP is for evaluation, not production)
- ❌ Custom model development (but we can provide guidance)
- ❌ Guaranteed fixes within specific timeframe

**Critical issues** will be prioritized and may get faster response.

---

**Q: Can I report the same issue multiple times?**

A: Please don't! **Check first**:

1. Is it in [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md)?
2. Did you already report it via email?
3. Is it in GitHub Issues (if enabled)?

**If yes to any**: We're already tracking it, no need to report again.

**If no**: Please report! We want to know about all issues.

---

## Timeline & Roadmap

### EAP Duration

**Q: How long will the EAP run?**

A: **Approximately 6-8 weeks** from EAP launch to GA preparation.

**Timeline**:
- **Week 1-2**: Onboarding, installation, initial exploration
- **Week 3-6**: Use case evaluation, custom model development
- **Week 7-8**: Final feedback, GA preparation

**Exact timeline** will be communicated via email.

---

**Q: When is the GA release?**

A: **Target**: IRIS 2026.1 (Q2 2026)

**Milestones**:
- EAP Launch: January 2025
- Feature Freeze: ~2 months before GA
- GA Release: Q2 2026

**Exact GA date** will be announced by InterSystems.

---

**Q: What happens between EAP and GA?**

A: **Iteration and improvement**:

1. **Bug fixes**: Fix critical bugs from EAP
2. **Documentation**: Fill gaps identified during EAP
3. **Performance**: Optimize based on EAP benchmarks
4. **Features**: Refine based on EAP feedback
5. **Testing**: Full QA testing before GA
6. **Docs integration**: Coordinate with docs.intersystems.com

**See [`EAP_ROADMAP.md`](EAP_ROADMAP.md) for complete roadmap.**

---

**Q: Can I influence the roadmap?**

A: **Absolutely!**

**Your input shapes**:
- Bug fix priorities
- Documentation improvements
- Feature priorities for GA
- Post-GA roadmap direction

**How to influence**:
- Complete surveys with detailed priorities
- Share your use cases and requirements
- Report what's working well (not just bugs!)
- Participate in optional feedback calls

**The more specific your feedback, the more actionable it is.**

---

## Production Use

### Using in Production

**Q: Can I use Custom Models in production during EAP?**

A: **No - evaluation only during EAP.**

**Reasons**:
- EAP is pre-release software
- API may change based on feedback
- Bugs may exist
- No SLA or production support

**After GA** (2026.1):
- ✅ Production use supported
- ✅ Stable API with backward compatibility
- ✅ Full documentation
- ✅ Standard support channels

**However**: You can **evaluate** production readiness during EAP:
- Test with realistic data volumes
- Review security considerations
- Assess operational requirements
- Plan production deployment

---

**Q: What should I consider for production deployment?**

A: **Key considerations** (evaluate during EAP):

**Security**:
- Model validation and testing
- Access control
- Data privacy

**Performance**:
- Prediction latency requirements
- Throughput requirements
- Hardware sizing

**Operations**:
- Monitoring and alerting
- Model update process
- Backup and recovery
- Troubleshooting procedures

**See [`deployment.md`](deployment.md) for current guidance. Production documentation will be expanded based on EAP feedback.**

---

**Q: Will my EAP models work in GA?**

A: **Goal**: Yes, with minimal changes.

**We aim for**:
- API compatibility between EAP and GA
- Clear migration guide for any breaking changes
- Deprecation warnings if API changes are needed

**However**: EAP is pre-release, so some changes may be necessary based on feedback.

**We will**:
- Communicate any breaking changes clearly
- Provide migration instructions
- Support you through any required updates

---

## After EAP

### GA Release

**Q: What happens when GA is released?**

A: **You get**:

1. **Stable release**:
   - Production-ready Custom Models
   - Stable API with backward compatibility guarantee
   - Full documentation on docs.intersystems.com

2. **Acknowledgment** (if desired):
   - Credit as EAP participant
   - Optional testimonial/case study opportunity
   - Recognition in GA announcement materials

3. **Continued access**:
   - Full access to GA release
   - Standard InterSystems support channels
   - Community forums and resources

---

**Q: Will I be acknowledged for EAP participation?**

A: **Yes, if you want to be!**

**We'd like to**:
- Acknowledge EAP participants in GA release notes
- Share use cases/testimonials (with permission)
- Recognize contributors who provided exceptional feedback

**Your choice**:
- Opt-in or opt-out via exit survey
- Choose level of recognition (name only, testimonial, case study, etc.)
- Review any materials before publication

**No pressure** - anonymous participation is fine too!

---

**Q: Can I continue providing feedback after GA?**

A: **Yes, please do!**

**Post-GA feedback channels**:
- Standard InterSystems support
- WRC (Worldwide Response Center)
- Community forums
- Feature requests via standard channels
- GitHub Issues (if enabled)

**Your EAP relationship** gives you a direct line to the product team for complex feedback.

---

### Future Enhancements

**Q: What features are coming after GA?**

A: **Under consideration** (based on EAP feedback):

- Timeseries model native support
- Model versioning and rollback
- Hot reload for model changes
- Model monitoring and drift detection
- Model marketplace/template library

**See [`EAP_ROADMAP.md#post-ga-future-considerations`](EAP_ROADMAP.md#post-ga-future-considerations) for full list.**

**Your EAP feedback will help prioritize these!**

---

**Q: Can I contribute to Custom Models after GA?**

A: **We're exploring community contribution options**:

**Potential areas**:
- Example models for specific industries
- Model templates and best practices
- Documentation improvements
- Tutorials and blog posts

**Details TBD** - we'll share more as we approach GA.

---

## Still Have Questions?

**Not answered here?**

- 📧 **Email**: thomas.dyar@intersystems.com
- 📚 **Documentation**: Check `docs/` directory
- 🔍 **Search**: GitHub repository for similar questions

**We'll update this FAQ** as new questions come up during the EAP!

---

**Thank you for participating in the EAP!**

**— The InterSystems Data Platforms Product Team**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-12
**Next Update**: Based on EAP questions

**Latest Version**: https://github.com/intersystems-community/integratedml-custom-models
