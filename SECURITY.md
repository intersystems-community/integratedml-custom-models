# Security Policy

## 🔒 IntegratedML Flexible Model Integration Framework Security

We take the security of the **IntegratedML Flexible Model Integration Framework** seriously. This document outlines our security policies, vulnerability reporting process, and best practices for secure usage.

## 🎯 Scope and Coverage

### 📦 In Scope
This security policy covers:

* **Core Framework Components** - Base models, database integration, configuration management
* **Demo Applications** - Credit Risk, Fraud Detection, Sales Forecasting, DNA Similarity demos
* **Database Integration** - InterSystems IRIS connectivity and IntegratedML integration
* **Docker Configuration** - Container setup and deployment scripts
* **Dependencies** - Third-party libraries and their security implications
* **Documentation** - Security guidance and best practices

### 🚫 Out of Scope
This policy does **not** cover:

* **InterSystems IRIS Platform** - Report security issues to InterSystems directly
* **Third-party ML Libraries** - Report to respective maintainers (scikit-learn, TensorFlow, etc.)
* **Infrastructure** - Your hosting environment, cloud providers, or production deployments
* **User Implementations** - Custom models or applications built using this framework

## 🛡️ Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Security Updates |
| ------- | ------------------ | ---------------- |
| 1.0.x   | ✅ Yes             | Active support   |
| 0.9.x   | ⚠️ Limited         | Critical fixes only |
| < 0.9   | ❌ No              | Not supported    |

**Note**: As a community-driven project, security updates are provided on a best-effort basis by volunteer maintainers.

## 🚨 Vulnerability Reporting

### 🔴 Critical Security Issues
For **critical security vulnerabilities** that could affect user data, system integrity, or enable unauthorized access:

**DO NOT** create public GitHub issues for security vulnerabilities.

Instead, please report them using **GitHub Security Advisories**:

1. **Go to**: [GitHub Security Advisories](../../security/advisories)
2. **Click**: "Report a vulnerability" 
3. **Provide**: Detailed information using the template below
4. **Wait**: For acknowledgment and coordination

### 🟡 Non-Critical Security Issues
For **general security improvements** or **potential issues** that don't pose immediate risk:

* Create a regular GitHub issue with the `security` label
* Use our [Bug Report template](../../issues/new?template=bug_report.md) 
* Mark the security implications clearly in your report

### 📋 Vulnerability Report Template

When reporting security issues, please include:

```
**Vulnerability Type**: [e.g., SQL Injection, Code Injection, Information Disclosure]

**Affected Components**: [e.g., Demo X, Database connector, Configuration parser]

**Attack Vector**: [Local/Network/Physical/Adjacent Network]

**Severity Assessment**: [Critical/High/Medium/Low]

**Description**: 
Clear description of the vulnerability and its potential impact

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Impact Assessment**:
- Data confidentiality: [High/Medium/Low]
- Data integrity: [High/Medium/Low] 
- System availability: [High/Medium/Low]
- Privilege escalation: [Yes/No]

**Proof of Concept**:
[Minimal code example demonstrating the issue - DO NOT include exploit code]

**Suggested Fix**:
[If you have ideas for addressing the vulnerability]

**Environment Details**:
- Framework version: 
- Python version:
- Operating system:
- IRIS version (if applicable):
- Docker setup: [Yes/No]

**Discoverer Information**:
- Name: [How you'd like to be credited]
- Organization: [Optional]
- Contact method: [For follow-up questions]
```

## ⏱️ Response Timeline

As a community-driven project, we aim for:

* **Acknowledgment**: Within 72 hours of report
* **Initial Assessment**: Within 1 week
* **Resolution Plan**: Within 2 weeks for critical issues
* **Patch Release**: Based on complexity and volunteer availability

**Note**: These are target timelines for volunteer-maintained projects and may vary based on maintainer availability and issue complexity.

## 🔧 Security Best Practices

### 🏗️ For Users and Implementers

#### 🗄️ Database Security
```python
# ✅ Good: Use connection parameters from secure configuration
iris_config = {
    'hostname': os.getenv('IRIS_HOST'),
    'port': int(os.getenv('IRIS_PORT')),
    'username': os.getenv('IRIS_USER'),
    'password': os.getenv('IRIS_PASSWORD'),  # Use secrets management
    'namespace': os.getenv('IRIS_NAMESPACE')
}

# ❌ Bad: Hard-coded credentials
iris_config = {
    'hostname': 'localhost',
    'username': 'admin',
    'password': 'password123'  # Never do this!
}
```

#### 🔐 Environment Configuration
```bash
# ✅ Good: Use environment variables for sensitive data
export IRIS_PASSWORD="your-secure-password"
export MODEL_API_KEY="your-api-key"

# ✅ Good: Set restrictive file permissions
chmod 600 .env
chmod 700 config/
```

#### 🐳 Docker Security
```yaml
# ✅ Good: Docker security practices
services:
  iris:
    # Use specific versions, not 'latest'
    image: intersystemsdc/iris-community:2023.1
    # Don't run as root
    user: "1000:1000"
    # Limit resources
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

#### 📊 Data Protection
```python
# ✅ Good: Validate and sanitize inputs
def validate_model_config(config):
    """Validate model configuration to prevent injection attacks."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Validate allowed keys
    allowed_keys = {'model_type', 'parameters', 'features'}
    if not set(config.keys()).issubset(allowed_keys):
        raise ValueError("Invalid configuration keys")
    
    return config

# ✅ Good: Use parameterized queries
def safe_database_query(connection, table_name, conditions):
    """Execute database query safely with parameters."""
    # Don't concatenate user input directly into SQL
    query = "SELECT * FROM ? WHERE conditions = ?"
    return connection.execute(query, [table_name, conditions])
```

### 🛠️ For Contributors

#### 🔍 Code Review Security Checklist
- [ ] **Input Validation**: All user inputs are validated and sanitized
- [ ] **SQL Injection**: Database queries use parameterized statements
- [ ] **Path Traversal**: File operations validate paths and prevent directory traversal
- [ ] **Code Injection**: Avoid `eval()`, `exec()`, or dynamic imports of user data
- [ ] **Information Disclosure**: No sensitive data in logs, error messages, or responses
- [ ] **Dependency Security**: New dependencies are from trusted sources and up-to-date

#### 🧪 Security Testing
```python
# Example: Security-focused test cases
def test_sql_injection_prevention():
    """Test that SQL injection attacks are prevented."""
    malicious_input = "'; DROP TABLE users; --"
    with pytest.raises(ValueError):
        execute_model_query(malicious_input)

def test_path_traversal_prevention():
    """Test that path traversal attacks are prevented."""
    malicious_path = "../../../etc/passwd"
    with pytest.raises(SecurityError):
        load_model_config(malicious_path)
```

## 🚧 Known Security Considerations

### ⚠️ Current Limitations
1. **Model Serialization**: Pickle-based model serialization can execute arbitrary code - only load trusted models
2. **Configuration Files**: YAML configuration files can execute code - validate inputs before parsing
3. **Database Connections**: Database credentials must be properly secured in production environments
4. **Docker Containers**: Default Docker setup is for development - harden for production use

### 🔄 Planned Improvements
- [ ] **Secure Model Serialization**: Migrate from pickle to safer serialization formats
- [ ] **Configuration Validation**: Enhanced YAML parsing with security restrictions
- [ ] **Audit Logging**: Comprehensive security event logging
- [ ] **Access Controls**: Role-based access control for model management
- [ ] **Encryption**: At-rest encryption for sensitive model parameters

## 🏆 Security Recognition

We appreciate security researchers and community members who help improve our security posture. Contributors who report valid security issues will be:

* **Credited** in security advisories (with permission)
* **Acknowledged** in release notes
* **Listed** in our security contributors section
* **Thanked** publicly for their responsible disclosure

### 🎖️ Security Contributors
*We'll list security contributors here as they help improve the project.*

## 📚 Security Resources

### 🔗 External Resources
* **OWASP Top 10**: [https://owasp.org/www-project-top-ten/](https://owasp.org/www-project-top-ten/)
* **Python Security**: [https://python-security.readthedocs.io/](https://python-security.readthedocs.io/)
* **Docker Security**: [https://docs.docker.com/engine/security/](https://docs.docker.com/engine/security/)
* **InterSystems Security**: [https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSA](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSA)

### 📖 Framework-Specific Guides
* [Production Deployment Security](docs/deployment/security.md) *(coming soon)*
* [Database Security Configuration](docs/database/security.md) *(coming soon)*
* [Model Security Best Practices](docs/models/security.md) *(coming soon)*

## 📞 Contact and Support

### 🆘 Emergency Contact
For **critical security issues** requiring immediate attention:
* Use GitHub Security Advisories (preferred)
* If GitHub is unavailable, contact active maintainers directly

### 💬 General Security Questions
For **non-urgent security questions**:
* Open a GitHub Discussion in the Security category
* Tag your issue with the `security` label
* Reference this security policy in your question

### 🤝 Security Collaboration
Interested in helping improve framework security?
* Review our [Contributing Guidelines](CONTRIBUTING.md)
* Join security-focused code reviews
* Contribute security documentation and examples
* Help with security testing and validation

---

**Last Updated**: August 27, 2025  
**Policy Version**: 1.0  
**Next Review**: February 27, 2026

*This security policy is adapted for community-driven open source projects and follows industry best practices for responsible disclosure and community security management.*