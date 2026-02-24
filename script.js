const terms = [
  {
    term: 'Algorithm',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A set of instructions a computer follows to solve a problem or make decisions.',
    learnMore:
      'In AI, algorithms define how data is processed and how predictions are produced. Different algorithms are chosen based on speed, interpretability, and accuracy needs.',
  },
  {
    term: 'Model',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A mathematical system trained on data to recognize patterns and make predictions.',
    learnMore:
      'After training, a model can generalize to new examples it has not seen before. Its quality depends on data quality, architecture, and tuning.',
  },
  {
    term: 'Training Data',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'Examples used to teach a machine learning model how to perform a task.',
    learnMore:
      'Training data should represent real-world conditions; otherwise, model performance can degrade when deployed to production.',
  },
  {
    term: 'Overfitting',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'When a model learns noise in training data and performs poorly on new data.',
    learnMore:
      'Overfitting is reduced using methods such as regularization, data augmentation, early stopping, and proper validation strategies.',
  },
  {
    term: 'Underfitting',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'When a model is too simple to capture patterns in the data.',
    learnMore:
      'Underfit models show poor performance on both training and test sets. Increasing model capacity or feature quality can help.',
  },
  {
    term: 'Hyperparameter',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'A configurable setting chosen before training, such as learning rate or batch size.',
    learnMore:
      'Hyperparameters are tuned through grid search, random search, or Bayesian optimization to improve model performance.',
  },
  {
    term: 'Gradient Descent',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'An optimization method that updates model weights to reduce prediction error.',
    learnMore:
      'It works by following the slope of the loss function. Variants include stochastic, mini-batch, and momentum-based approaches.',
  },
  {
    term: 'Neural Network',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A model inspired by the brain, built from layers of interconnected computation units.',
    learnMore:
      'Deep neural networks can learn highly complex patterns from images, audio, and text by stacking many hidden layers.',
  },
  {
    term: 'Transformer',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A neural architecture that uses attention to process relationships in sequences efficiently.',
    learnMore:
      'Transformers replaced many recurrent models and now power modern language models because they scale well with data and compute.',
  },
  {
    term: 'Attention Mechanism',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A technique allowing models to focus on relevant parts of input when generating output.',
    learnMore:
      'Self-attention calculates dependencies between tokens directly, enabling better context handling in long sequences.',
  },
  {
    term: 'Large Language Model (LLM)',
    category: 'Generative AI',
    level: 'Beginner',
    definition: 'A transformer-based model trained on vast text corpora to understand and generate language.',
    learnMore:
      'LLMs can perform translation, summarization, coding, and question answering through prompting and optional fine-tuning.',
  },
  {
    term: 'Prompt Engineering',
    category: 'Generative AI',
    level: 'Intermediate',
    definition: 'Designing effective prompts to guide model behavior and output quality.',
    learnMore:
      'Strategies include role prompting, few-shot examples, constraints, and structured output formats for reliability.',
  },
  {
    term: 'Fine-Tuning',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'Further training a pre-trained model on domain-specific data for specialized tasks.',
    learnMore:
      'Fine-tuning can improve performance in narrow domains, but it requires careful evaluation to avoid overfitting or drift.',
  },
  {
    term: 'Embeddings',
    category: 'NLP',
    level: 'Intermediate',
    definition: 'Numeric vector representations that encode semantic meaning of words, sentences, or documents.',
    learnMore:
      'Similar meanings map to nearby vectors, making embeddings useful for search, recommendation, clustering, and retrieval.',
  },
  {
    term: 'Tokenization',
    category: 'NLP',
    level: 'Beginner',
    definition: 'The process of splitting text into units such as words, subwords, or symbols.',
    learnMore:
      'Tokenizer design affects model context length, speed, and multilingual performance. Subword tokenizers balance vocabulary size and coverage.',
  },
  {
    term: 'Named Entity Recognition (NER)',
    category: 'NLP',
    level: 'Intermediate',
    definition: 'Identifying and classifying entities like people, locations, and organizations in text.',
    learnMore:
      'NER supports information extraction pipelines in legal, healthcare, and customer support use cases.',
  },
  {
    term: 'Sentiment Analysis',
    category: 'NLP',
    level: 'Beginner',
    definition: 'Determining emotional tone in text, such as positive, negative, or neutral.',
    learnMore:
      'Businesses use sentiment models to monitor product feedback, social media, and support conversations at scale.',
  },
  {
    term: 'BLEU Score',
    category: 'NLP',
    level: 'Advanced',
    definition: 'A metric for evaluating machine translation quality by comparing output to reference translations.',
    learnMore:
      'BLEU estimates n-gram overlap but may miss semantic quality, so it is often combined with human or learned evaluations.',
  },
  {
    term: 'Convolutional Neural Network (CNN)',
    category: 'Computer Vision',
    level: 'Intermediate',
    definition: 'A neural network using convolution filters to detect patterns in images.',
    learnMore:
      'CNNs exploit spatial structure and have been foundational for classification, segmentation, and detection tasks.',
  },
  {
    term: 'Object Detection',
    category: 'Computer Vision',
    level: 'Intermediate',
    definition: 'Detecting objects in images and drawing bounding boxes around them.',
    learnMore:
      'Detection systems power applications like autonomous driving, retail analytics, and robotics perception.',
  },
  {
    term: 'Image Segmentation',
    category: 'Computer Vision',
    level: 'Advanced',
    definition: 'Assigning a label to each pixel in an image to separate objects and regions.',
    learnMore:
      'Segmentation is used in medical imaging, satellite mapping, and autonomous systems where precise boundaries matter.',
  },
  {
    term: 'Transfer Learning',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'Reusing a model trained on one task as a starting point for another related task.',
    learnMore:
      'Transfer learning reduces required data and compute, especially useful when labeled data is limited.',
  },
  {
    term: 'Reinforcement Learning',
    category: 'Machine Learning',
    level: 'Advanced',
    definition: 'A learning framework where agents optimize actions through rewards and penalties.',
    learnMore:
      'RL is used in game playing, robotics, and control systems where decisions impact future outcomes.',
  },
  {
    term: 'Hallucination',
    category: 'Generative AI',
    level: 'Intermediate',
    definition: 'When a model generates plausible-sounding but incorrect or fabricated information.',
    learnMore:
      'Mitigation includes retrieval grounding, confidence checks, prompt constraints, and human review loops.',
  },
  {
    term: 'Retrieval-Augmented Generation (RAG)',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A method that retrieves external knowledge and feeds it to a model before generation.',
    learnMore:
      'RAG improves factuality and freshness by grounding responses in trusted documents rather than only model memory.',
  },
  {
    term: 'Bias',
    category: 'Ethics & Safety',
    level: 'Beginner',
    definition: 'Systematic unfairness in model outputs due to skewed data, assumptions, or design choices.',
    learnMore:
      'Bias can cause discriminatory outcomes, so audits, balanced datasets, and fairness metrics are critical.',
  },
  {
    term: 'Fairness',
    category: 'Ethics & Safety',
    level: 'Intermediate',
    definition: 'The principle that AI systems should treat individuals and groups equitably.',
    learnMore:
      'Fairness has multiple definitions (e.g., equalized odds, demographic parity), often requiring tradeoff decisions.',
  },
  {
    term: 'Explainability',
    category: 'Ethics & Safety',
    level: 'Intermediate',
    definition: 'The ability to understand and communicate why a model made a specific decision.',
    learnMore:
      'Techniques include feature importance, attention visualization, SHAP values, and model cards.',
  },
  {
    term: 'Adversarial Attack',
    category: 'Ethics & Safety',
    level: 'Advanced',
    definition: 'An intentional input manipulation designed to make a model produce wrong outputs.',
    learnMore:
      'Even tiny perturbations can fool vision models, motivating robust training and threat modeling.',
  },
  {
    term: 'Alignment',
    category: 'Ethics & Safety',
    level: 'Advanced',
    definition: 'Ensuring AI behavior matches human values, intentions, and safety constraints.',
    learnMore:
      'Alignment research addresses harmful outputs, objective misspecification, and long-term control challenges.',
  },
  {
    term: 'Data Drift',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'A shift in real-world data patterns over time compared with training data.',
    learnMore:
      'Drift monitoring is essential in production systems and often triggers retraining or pipeline updates.',
  },
  {
    term: 'Precision and Recall',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'Metrics that balance false positives and false negatives in classification tasks.',
    learnMore:
      'Precision asks, “Of predicted positives, how many were correct?” Recall asks, “Of true positives, how many were found?”',
  },
];

const categories = ['All', ...new Set(terms.map((item) => item.category))];
const exploredTerms = new Set();

const cardsGrid = document.getElementById('cardsGrid');
const categoryFilters = document.getElementById('categoryFilters');
const searchInput = document.getElementById('searchInput');
const progressTracker = document.getElementById('progressTracker');
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
const randomBtn = document.getElementById('randomBtn');
const emptyState = document.getElementById('emptyState');

let activeCategory = 'All';

function levelClass(level) {
  return level.toLowerCase();
}

function updateProgress(totalVisible = terms.length) {
  progressTracker.textContent = `${exploredTerms.size} of ${totalVisible} terms explored`;
}

function createCard(item) {
  const card = document.createElement('article');
  card.className = 'term-card';
  card.setAttribute('tabindex', '0');

  card.innerHTML = `
    <div class="term-head">
      <h3 class="term-name">${item.term}</h3>
      <span class="badge ${levelClass(item.level)}">${item.level}</span>
    </div>
    <p class="category">${item.category}</p>
    <p class="definition">${item.definition}</p>
    <p class="learn-more"><strong>Learn More:</strong> ${item.learnMore}</p>
  `;

  const toggleExpand = () => {
    card.classList.toggle('expanded');
    exploredTerms.add(item.term);
    updateProgress();
  };

  card.addEventListener('click', toggleExpand);
  card.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleExpand();
    }
  });

  return card;
}

function renderCategories() {
  categoryFilters.innerHTML = '';
  categories.forEach((category) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-btn ${category === activeCategory ? 'active' : ''}`;
    btn.textContent = category;

    btn.addEventListener('click', () => {
      activeCategory = category;
      renderCategories();
      renderCards();
    });

    categoryFilters.appendChild(btn);
  });
}

function getFilteredTerms() {
  const query = searchInput.value.trim().toLowerCase();

  return terms.filter((item) => {
    const inCategory = activeCategory === 'All' || item.category === activeCategory;
    const inSearch =
      item.term.toLowerCase().includes(query) ||
      item.definition.toLowerCase().includes(query) ||
      item.category.toLowerCase().includes(query);

    return inCategory && inSearch;
  });
}

function renderCards() {
  const filtered = getFilteredTerms();
  cardsGrid.innerHTML = '';

  filtered.forEach((item) => {
    cardsGrid.appendChild(createCard(item));
  });

  emptyState.classList.toggle('hidden', filtered.length > 0);
  updateProgress(filtered.length || 0);
}

function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('theme', mode);
  themeLabel.textContent = mode === 'dark' ? '☀️ Light' : '🌙 Dark';
}

function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) {
    setTheme(saved);
    return;
  }

  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(systemDark ? 'dark' : 'light');
}

function highlightRandomCard() {
  const cards = Array.from(document.querySelectorAll('.term-card'));
  if (!cards.length) return;

  const randomCard = cards[Math.floor(Math.random() * cards.length)];
  randomCard.classList.add('highlight');
  randomCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

  if (!randomCard.classList.contains('expanded')) {
    randomCard.classList.add('expanded');
  }

  const term = randomCard.querySelector('.term-name')?.textContent;
  if (term) {
    exploredTerms.add(term);
    updateProgress(cards.length);
  }

  setTimeout(() => {
    randomCard.classList.remove('highlight');
  }, 1200);
}

searchInput.addEventListener('input', renderCards);
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  setTheme(next);
});
randomBtn.addEventListener('click', highlightRandomCard);

initTheme();
renderCategories();
renderCards();
