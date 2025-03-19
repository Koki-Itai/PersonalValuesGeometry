schwartz_values_instructions = """#### **Self-Direction**
- **Defining goal**: Independent thought and action—choosing, creating, exploring.
- **Description**: Derived from needs for control, mastery, autonomy, and independence.
- **KeyWords**: Creativity, freedom, choosing own goals, curious, independent, self-respect, intelligent, privacy.

#### **Stimulation**
- **Defining goal**: Excitement, novelty, and challenge in life.
- **Description**: Based on the need for variety and stimulation to maintain an optimal level of activation.
- **KeyWords**: A varied life, an exciting life, daring.

#### **Hedonism**
- **Defining goal**: Pleasure or sensuous gratification for oneself.
- **Description**: Rooted in organismic needs and pleasure associated with their satisfaction.
- **KeyWords**: Pleasure, enjoying life, self-indulgent.

#### **Achievement**
- **Defining goal**: Personal success through demonstrating competence according to social standards.
- **Description**: Emphasizes performance in terms of cultural standards to gain social approval.
- **KeyWords**: Ambitious, successful, capable, influential, intelligent, self-respect, social recognition.

#### **Power**
- **Defining goal**: Social status and prestige, control or dominance over people and resources.
- **Description**: Reflects social status differentiation and individual needs for dominance.
- **KeyWords**: Authority, wealth, social power, preserving public image, social recognition.

#### **Security**
- **Defining goal**: Safety, harmony, and stability of society, relationships, and self.
- **Description**: Ensures group and individual survival; includes personal and societal security.
- **KeyWords**: Social order, family security, national security, clean, reciprocation of favors, healthy, moderate, belonging.

#### **Conformity**
- **Defining goal**: Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms.
- **Description**: Encourages individuals to inhibit actions that disrupt group harmony.
- **KeyWords**: Obedient, self-discipline, politeness, honoring parents and elders, loyal, responsible.

#### **Tradition**
- **Defining goal**: Respect, commitment, and acceptance of the customs and ideas provided by one's culture or religion.
- **Description**: Emphasizes subordinating oneself to established cultural or religious customs.
- **KeyWords**: Respect for tradition, humble, devout, accepting one's portion in life, moderate, spiritual life.

#### **Benevolence**
- **Defining goal**: Preserving and enhancing the welfare of those with whom one is in frequent personal contact (the ‘in-group’).
- **Description**: Promotes voluntary concern for the well-being of close others.
- **KeyWords**: Helpful, honest, forgiving, responsible, loyal, true friendship, mature love, belonging, meaning in life, spiritual life.

#### **Universalism**
- **Defining goal**: Understanding, appreciation, tolerance, and protection for the welfare of all people and nature.
- **Description**: Extends beyond the in-group, emphasizing equality, justice, and environmental protection.
- **KeyWords**: Broadminded, social justice, equality, world at peace, world of beauty, unity with nature, wisdom, protecting the environment, inner harmony, spiritual life.
"""

PROMPT_TEMPLATES = {
    "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
    "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
    "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
    "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
    "bare": "{}",
    "theme": "What is the main theme in this text: {}\nKey themes include:",
    "topic": "What is the main topic in this text: {}\nMain topic covered are:",
    "explicit_schwartz": "What values are represented in following text. Values are defined based on Schwartz's Value Theory. \n\n# **Schwartz Value Definitions**\n\n{}\n\n== Let's begin ! ===\n# Target text\n{}\n\nThe text demonstrates values of:",
}
