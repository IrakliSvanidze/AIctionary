let currentLang = 'en';

const ui = {
  en: {
    eyebrow: 'Glossary',
    title: 'AI Dictionary',
    subtitle: 'Your guide to understanding Artificial Intelligence',
    searchPlaceholder: 'Search terms, definitions, or categories...',
    randomBtn: '🎲 Random Term',
    themeDark: '🌙 Dark',
    themeLight: '☀️ Light',
    learnMoreLabel: 'Learn More',
    empty: 'No terms found. Try another search or category.',
    progressOf: 'of',
    progressLabel: 'terms explored',
  },
  geo: {
    eyebrow: 'გლოსარიუმი',
    title: 'AI ლექსიკონი',
    subtitle: 'თქვენი სახელმძღვანელო ხელოვნური ინტელექტის გასაგებად',
    searchPlaceholder: 'მოძებნეთ ტერმინები, განმარტებები ან კატეგორიები...',
    randomBtn: '🎲 შემთხვევითი ტერმინი',
    themeDark: '🌙 მუქი',
    themeLight: '☀️ ნათელი',
    learnMoreLabel: 'გაიგეთ მეტი',
    empty: 'ტერმინები ვერ მოიძებნა. სცადეთ სხვა ძიება ან კატეგორია.',
    progressOf: '/',
    progressLabel: 'ტერმინი შესწავლილია',
  },
};

const categoryNames = {
  en: {
    All: 'All',
    'Machine Learning': 'Machine Learning',
    'Generative AI': 'Generative AI',
    NLP: 'NLP',
    'Computer Vision': 'Computer Vision',
    'Ethics & Safety': 'Ethics & Safety',
  },
  geo: {
    All: 'ყველა',
    'Machine Learning': 'მანქანური სწავლება',
    'Generative AI': 'გენერაციული AI',
    NLP: 'NLP',
    'Computer Vision': 'კომპიუტერული მხედველობა',
    'Ethics & Safety': 'ეთიკა და უსაფრთხოება',
  },
};

const levelNames = {
  en: { Beginner: 'Beginner', Intermediate: 'Intermediate', Advanced: 'Advanced' },
  geo: { Beginner: 'დამწყები', Intermediate: 'საშუალო', Advanced: 'მოწინავე' },
};

const terms = [
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Algorithm', definition: 'A set of instructions a computer follows to solve a problem or make decisions.', learnMore: 'In AI, algorithms define how data is processed and how predictions are produced. Different algorithms are chosen based on speed, interpretability, and accuracy needs.' },
    geo: { term: 'ალგორითმი', definition: 'ინსტრუქციების ნაკრები, რომელსაც კომპიუტერი მიჰყვება პრობლემის გადასაჭრელად ან გადაწყვეტილებების მისაღებად.', learnMore: 'AI-ში ალგორითმები განსაზღვრავს მონაცემთა დამუშავების და პროგნოზების გამომუშავების მეთოდს. სხვადასხვა ალგორითმი შეირჩევა სიჩქარის, ინტერპრეტირებადობისა და სიზუსტის მიხედვით.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Model', definition: 'A mathematical system trained on data to recognize patterns and make predictions.', learnMore: 'After training, a model can generalize to new examples it has not seen before. Its quality depends on data quality, architecture, and tuning.' },
    geo: { term: 'მოდელი', definition: 'მათემატიკური სისტემა, რომელიც გამოიყენება მონაცემებში პატერნების ამოსაცნობად და პროგნოზების გასაკეთებლად.', learnMore: 'გაწვრთნის შემდეგ მოდელი ახლის მაგალითებზე განაზოგადებს. ხარისხი დამოკიდებულია მონაცემებზე, არქიტექტურასა და კონფიგურაციაზე.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Training Data', definition: 'Examples used to teach a machine learning model how to perform a task.', learnMore: 'Training data should represent real-world conditions; otherwise, model performance can degrade when deployed to production.' },
    geo: { term: 'სასწავლო მონაცემები', definition: 'მაგალითები, რომლებიც გამოიყენება მანქანური სწავლების მოდელის გასაწვრთნელად.', learnMore: 'სასწავლო მონაცემები უნდა ასახავდეს რეალურ პირობებს; წინააღმდეგ შემთხვევაში, მოდელის მუშაობა გაუარესდება პრაქტიკაში.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Overfitting', definition: 'When a model learns noise in training data and performs poorly on new data.', learnMore: 'Overfitting is reduced using methods such as regularization, data augmentation, early stopping, and proper validation strategies.' },
    geo: { term: 'გადამეტებული მორგება', definition: 'მდგომარეობა, როდესაც მოდელი ზედმეტად ეგუება სასწავლო მონაცემებს და ახალ მონაცემებზე ცუდ შედეგს იძლევა.', learnMore: 'გამოსასწორებლად გამოიყენება რეგულარიზაცია, მონაცემთა გამდიდრება, ადრეული გაჩერება და სათანადო ვალიდაციის სტრატეგიები.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Underfitting', definition: 'When a model is too simple to capture patterns in the data.', learnMore: 'Underfit models show poor performance on both training and test sets. Increasing model capacity or feature quality can help.' },
    geo: { term: 'არასაკმარისი მორგება', definition: 'მდგომარეობა, როდესაც მოდელი ზედმეტად მარტივია მონაცემებში პატერნების გამოვლენისათვის.', learnMore: 'ასეთი მოდელები ცუდ შედეგს იძლევა სასწავლო და სატესტო კრებულებზე. გამოსავალია მოდელის სიმძლავრის ან ნიშნების ხარისხის გაზრდა.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Hyperparameter', definition: 'A configurable setting chosen before training, such as learning rate or batch size.', learnMore: 'Hyperparameters are tuned through grid search, random search, or Bayesian optimization to improve model performance.' },
    geo: { term: 'ჰიპერპარამეტრი', definition: 'კონფიგურირებადი პარამეტრი, რომელიც ირჩევა გაწვრთნამდე, მაგ. სწავლის სიჩქარე ან ბატჩის ზომა.', learnMore: 'ჰიპერპარამეტრები მორგებულია Grid Search-ის, Random Search-ის ან ბაიზური ოპტიმიზაციის გამოყენებით.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Gradient Descent', definition: 'An optimization method that updates model weights to reduce prediction error.', learnMore: 'It works by following the slope of the loss function. Variants include stochastic, mini-batch, and momentum-based approaches.' },
    geo: { term: 'გრადიენტის დაშვება', definition: 'ოპტიმიზაციის მეთოდი, რომელიც ამცირებს პროგნოზის შეცდომას მოდელის წონების განახლებით.', learnMore: 'მეთოდი მიჰყვება დანაკარგის ფუნქციის დახრილობას. ვარიანტებია სტოქასტური, მინი-ბატჩური და იმპულსზე დაფუძნებული მიდგომები.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Neural Network', definition: 'A model inspired by the brain, built from layers of interconnected computation units.', learnMore: 'Deep neural networks can learn highly complex patterns from images, audio, and text by stacking many hidden layers.' },
    geo: { term: 'ნეირონული ქსელი', definition: 'ტვინზე შთაგონებული მოდელი, რომელიც შედგება ურთიერთდაკავშირებული გამოთვლითი ერთეულების შრეებისაგან.', learnMore: 'ღრმა ნეირონულ ქსელებს შეუძლიათ ისწავლონ რთული პატერნები სურათებიდან, აუდიოდან და ტექსტიდან მრავალი ფარული შრის გამოყენებით.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Transformer', definition: 'A neural architecture that uses attention to process relationships in sequences efficiently.', learnMore: 'Transformers replaced many recurrent models and now power modern language models because they scale well with data and compute.' },
    geo: { term: 'ტრანსფორმერი', definition: 'ნეირონული არქიტექტურა, რომელიც ყურადღების მექანიზმის გამოყენებით ამუშავებს თანმიმდევრობებს.', learnMore: 'ტრანსფორმერებმა ჩაანაცვლა მრავალი რეკურენტული მოდელი და ახლა უდევს საფუძვლად თანამედროვე ენობრივ მოდელებს.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Attention Mechanism', definition: 'A technique allowing models to focus on relevant parts of input when generating output.', learnMore: 'Self-attention calculates dependencies between tokens directly, enabling better context handling in long sequences.' },
    geo: { term: 'ყურადღების მექანიზმი', definition: 'ტექნიკა, რომელიც საშუალებას აძლევს მოდელს გამომუშავებისას ყურადღება გაამახვილოს შეყვანის რელევანტურ ნაწილებზე.', learnMore: 'თვით-ყურადღება ითვლის ტოკენებს შორის დამოკიდებულებებს პირდაპირ, რაც გრძელ თანმიმდევრობებში კონტექსტის უკეთ დამუშავებას უზრუნველყოფს.' },
  },
  {
    category: 'Generative AI', level: 'Beginner',
    en: { term: 'Large Language Model (LLM)', definition: 'A transformer-based model trained on vast text corpora to understand and generate language.', learnMore: 'LLMs can perform translation, summarization, coding, and question answering through prompting and optional fine-tuning.' },
    geo: { term: 'დიდი ენობრივი მოდელი (LLM)', definition: 'ტრანსფორმერზე დაფუძნებული მოდელი, რომელიც ტექსტის უზარმაზარ კორპუსზეა გაწვრთნილი ენის გასაგებად და გენერირებისათვის.', learnMore: 'LLM-ებს შეუძლიათ შეასრულონ თარგმანი, შეჯამება, დაპროგრამება და კითხვა-პასუხი პრომპტინგისა და ოპციური ზუსტი მორგების გამოყენებით.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Prompt Engineering', definition: 'Designing effective prompts to guide model behavior and output quality.', learnMore: 'Strategies include role prompting, few-shot examples, constraints, and structured output formats for reliability.' },
    geo: { term: 'პრომპტ-ინჟინერია', definition: 'ეფექტური პრომპტების შექმნა მოდელის ქცევისა და გამომუშავების ხარისხის სამართავად.', learnMore: 'სტრატეგიებია: როლ-პრომპტინგი, მცირე ნიმუშები, შეზღუდვები და სტრუქტურული გამომუშავების ფორმატები.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Fine-Tuning', definition: 'Further training a pre-trained model on domain-specific data for specialized tasks.', learnMore: 'Fine-tuning can improve performance in narrow domains, but it requires careful evaluation to avoid overfitting or drift.' },
    geo: { term: 'ზუსტი მორგება', definition: 'წინასწარ გაწვრთნილი მოდელის დამატებითი ვარჯიში დარგობრივ მონაცემებზე სპეციალიზებული ამოცანებისთვის.', learnMore: 'ზუსტი მორგება აუმჯობესებს შედეგს ვიწრო სფეროში, მაგრამ მოითხოვს ყურადღებით შეფასებას გადამეტებული მორგების ან გადახრის თავიდან ასაცილებლად.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Embeddings', definition: 'Numeric vector representations that encode semantic meaning of words, sentences, or documents.', learnMore: 'Similar meanings map to nearby vectors, making embeddings useful for search, recommendation, clustering, and retrieval.' },
    geo: { term: 'ვექტორული წარმოდგენები', definition: 'რიცხვითი ვექტორული წარმოდგენები, რომლებიც კოდირებს სიტყვების, წინადადებების ან დოკუმენტების სემანტიკურ მნიშვნელობას.', learnMore: 'მსგავსი მნიშვნელობები ახლო ვექტორებზე გადაისახება, რაც ვექტორებს სასარგებლოს ხდის ძიებისთვის, რეკომენდაციებისთვის და კლასტერიზაციისთვის.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Tokenization', definition: 'The process of splitting text into units such as words, subwords, or symbols.', learnMore: 'Tokenizer design affects model context length, speed, and multilingual performance. Subword tokenizers balance vocabulary size and coverage.' },
    geo: { term: 'ტოკენიზაცია', definition: 'ტექსტის სიტყვებად, ქვე-სიტყვებად ან სიმბოლოებად დაყოფის პროცესი.', learnMore: 'ტოკენიზატორის დიზაინი გავლენას ახდენს მოდელის კონტექსტის სიგრძეზე, სიჩქარესა და მრავალენოვან შედეგებზე.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Named Entity Recognition (NER)', definition: 'Identifying and classifying entities like people, locations, and organizations in text.', learnMore: 'NER supports information extraction pipelines in legal, healthcare, and customer support use cases.' },
    geo: { term: 'სახელდებული ობიექტების ამოცნობა (NER)', definition: 'ტექსტში ადამიანების, ადგილების და ორგანიზაციების ამოცნობა და კლასიფიკაცია.', learnMore: 'NER მხარს უჭერს ინფორმაციის ამოღების პროცესებს სამართლის, ჯანდაცვისა და კლიენტთა მომსახურების სფეროებში.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Sentiment Analysis', definition: 'Determining emotional tone in text, such as positive, negative, or neutral.', learnMore: 'Businesses use sentiment models to monitor product feedback, social media, and support conversations at scale.' },
    geo: { term: 'სენტიმენტის ანალიზი', definition: 'ტექსტში ემოციური ტონის განსაზღვრა: დადებითი, უარყოფითი ან ნეიტრალური.', learnMore: 'კომპანიები სენტიმენტის მოდელებს იყენებს პროდუქტის გამოხმაურებისა და სოციალური მედიის მონიტორინგისთვის.' },
  },
  {
    category: 'NLP', level: 'Advanced',
    en: { term: 'BLEU Score', definition: 'A metric for evaluating machine translation quality by comparing output to reference translations.', learnMore: 'BLEU estimates n-gram overlap but may miss semantic quality, so it is often combined with human or learned evaluations.' },
    geo: { term: 'BLEU ქულა', definition: 'მეტრიკა მანქანური თარგმანის ხარისხის შესაფასებლად, რომელიც შედეგს საცნობარო თარგმანებს ადარებს.', learnMore: 'BLEU აფასებს n-გრამების გადაფარვას, თუმცა შეიძლება გამოტოვოს სემანტიკური ხარისხი, ამიტომ ხშირად ერთვება ადამიანური შეფასება.' },
  },
  {
    category: 'Computer Vision', level: 'Intermediate',
    en: { term: 'Convolutional Neural Network (CNN)', definition: 'A neural network using convolution filters to detect patterns in images.', learnMore: 'CNNs exploit spatial structure and have been foundational for classification, segmentation, and detection tasks.' },
    geo: { term: 'კონვოლუციური ნეირონული ქსელი (CNN)', definition: 'ნეირონული ქსელი, რომელიც კონვოლუციის ფილტრების გამოყენებით გამოსახულებებში პატერნებს ავლენს.', learnMore: 'CNN-ები სივრცულ სტრუქტურას იყენებს და საფუძვლობრივია კლასიფიკაციის, სეგმენტაციისა და გამოვლენის ამოცანებისთვის.' },
  },
  {
    category: 'Computer Vision', level: 'Intermediate',
    en: { term: 'Object Detection', definition: 'Detecting objects in images and drawing bounding boxes around them.', learnMore: 'Detection systems power applications like autonomous driving, retail analytics, and robotics perception.' },
    geo: { term: 'ობიექტის გამოვლენა', definition: 'გამოსახულებებში ობიექტების აღმოჩენა და მათ გარშემო შემოსაზღვრელი ჩარჩოების დახატვა.', learnMore: 'გამოვლენის სისტემები გამოიყენება ავტომატური მართვის, საცალო ვაჭრობის ანალიტიკისა და რობოტიკის სფეროებში.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Image Segmentation', definition: 'Assigning a label to each pixel in an image to separate objects and regions.', learnMore: 'Segmentation is used in medical imaging, satellite mapping, and autonomous systems where precise boundaries matter.' },
    geo: { term: 'გამოსახულების სეგმენტაცია', definition: 'გამოსახულების თითოეულ პიქსელს ეტიკეტის მიმინიჭება ობიექტებისა და რეგიონების გამოსაყოფად.', learnMore: 'სეგმენტაცია გამოიყენება სამედიცინო გამოსახულებებში, სატელიტურ რუკებსა და ავტონომიურ სისტემებში.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Transfer Learning', definition: 'Reusing a model trained on one task as a starting point for another related task.', learnMore: 'Transfer learning reduces required data and compute, especially useful when labeled data is limited.' },
    geo: { term: 'გადაცემითი სწავლება', definition: 'ერთ ამოცანაზე გაწვრთნილი მოდელის სხვა, მასთან დაკავშირებული ამოცანისთვის გამოყენება.', learnMore: 'გადაცემითი სწავლება ამცირებს საჭირო მონაცემების და გამოთვლის მოცულობას, განსაკუთრებით მაშინ, როდესაც მონიშნული მონაცემები შეზღუდულია.' },
  },
  {
    category: 'Machine Learning', level: 'Advanced',
    en: { term: 'Reinforcement Learning', definition: 'A learning framework where agents optimize actions through rewards and penalties.', learnMore: 'RL is used in game playing, robotics, and control systems where decisions impact future outcomes.' },
    geo: { term: 'განმამტკიცებელი სწავლება', definition: 'სწავლების ჩარჩო, სადაც აგენტები ოპტიმიზაციას ახდენს ჯილდოებისა და სასჯელების გამოყენებით.', learnMore: 'გამოიყენება თამაშებში, რობოტიკასა და საკონტროლო სისტემებში, სადაც გადაწყვეტილებები მომავალ შედეგებზე მოქმედებს.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Hallucination', definition: 'When a model generates plausible-sounding but incorrect or fabricated information.', learnMore: 'Mitigation includes retrieval grounding, confidence checks, prompt constraints, and human review loops.' },
    geo: { term: 'ჰალუცინაცია', definition: 'მდგომარეობა, როდესაც მოდელი ამყარებს სწორი ჟღერადობის, მაგრამ მცდარ ან გამოგონილ ინფორმაციას.', learnMore: 'შემარბილებელი ზომებია: მოძიებაზე დაფუძნება, სანდოობის შემოწმება, პრომპტის შეზღუდვები და ადამიანური გადამოწმება.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Retrieval-Augmented Generation (RAG)', definition: 'A method that retrieves external knowledge and feeds it to a model before generation.', learnMore: 'RAG improves factuality and freshness by grounding responses in trusted documents rather than only model memory.' },
    geo: { term: 'მოძიებით გამდიდრებული გენერაცია (RAG)', definition: 'მეთოდი, რომელიც გარე ცოდნას მოიპოვებს და გენერაციამდე მოდელს გადასცემს.', learnMore: 'RAG აუმჯობესებს სიზუსტესა და სიახლეს პასუხების სანდო დოკუმენტებზე დაყრდნობით, მხოლოდ მოდელის მეხსიერების ნაცვლად.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'Bias', definition: 'Systematic unfairness in model outputs due to skewed data, assumptions, or design choices.', learnMore: 'Bias can cause discriminatory outcomes, so audits, balanced datasets, and fairness metrics are critical.' },
    geo: { term: 'მიკერძოება', definition: 'სისტემატური უსამართლობა მოდელის გამომუშავებაში, რომელიც გამოწვეულია გადახრილი მონაცემებით, ვარაუდებით ან დიზაინის გადაწყვეტილებებით.', learnMore: 'მიკერძოებამ შეიძლება გამოიწვიოს დისკრიმინაციული შედეგები, ამიტომ კრიტიკულია აუდიტი, დაბალანსებული მონაცემები და სამართლიანობის მეტრიკები.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Fairness', definition: 'The principle that AI systems should treat individuals and groups equitably.', learnMore: 'Fairness has multiple definitions (e.g., equalized odds, demographic parity), often requiring tradeoff decisions.' },
    geo: { term: 'სამართლიანობა', definition: 'პრინციპი, რომ AI სისტემებმა ინდივიდები და ჯგუფები თანასწორად უნდა მოეპყრათ.', learnMore: 'სამართლიანობას მრავალი განმარტება აქვს (მაგ. თანაბარი შანსები, დემოგრაფიული პარიტეტი), რაც ხშირად კომპრომისულ გადაწყვეტილებებს საჭიროებს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Explainability', definition: 'The ability to understand and communicate why a model made a specific decision.', learnMore: 'Techniques include feature importance, attention visualization, SHAP values, and model cards.' },
    geo: { term: 'ახსნადობა', definition: 'შესაძლებლობა გავიგოთ და ავხსნათ, რატომ მიიღო მოდელმა კონკრეტული გადაწყვეტილება.', learnMore: 'ტექნიკებია: ნიშნების მნიშვნელობა, ყურადღების ვიზუალიზაცია, SHAP მნიშვნელობები და მოდელის ბარათები.' },
  },
  {
    category: 'Ethics & Safety', level: 'Advanced',
    en: { term: 'Adversarial Attack', definition: 'An intentional input manipulation designed to make a model produce wrong outputs.', learnMore: 'Even tiny perturbations can fool vision models, motivating robust training and threat modeling.' },
    geo: { term: 'მოწინააღმდეგური შეტევა', definition: 'შეყვანის განზრახ მანიპულაცია, რომელიც მოდელს არასწორი შედეგის გამოსამუშავებლად არის შექმნილი.', learnMore: 'მცირე ჩარევებმაც კი შეიძლება მოატყუოს მხედველობის მოდელები, რაც მდგრადი გაწვრთნის და საფრთხის მოდელირების მოტივაციას წარმოადგენს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Advanced',
    en: { term: 'Alignment', definition: 'Ensuring AI behavior matches human values, intentions, and safety constraints.', learnMore: 'Alignment research addresses harmful outputs, objective misspecification, and long-term control challenges.' },
    geo: { term: 'გასწორება', definition: 'AI-ის ქცევის ადამიანური ღირებულებებთან, განზრახვებთან და უსაფრთხოების შეზღუდვებთან შესაბამისობის უზრუნველყოფა.', learnMore: 'გასწორების კვლევა ეხება მავნე გამომუშავებებს, მიზნის არასწორ სპეციფიკაციას და გრძელვადიანი კონტროლის გამოწვევებს.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Data Drift', definition: 'A shift in real-world data patterns over time compared with training data.', learnMore: 'Drift monitoring is essential in production systems and often triggers retraining or pipeline updates.' },
    geo: { term: 'მონაცემთა გადახრა', definition: 'რეალური სამყაროს მონაცემების პატერნების ცვლილება დროთა განმავლობაში სასწავლო მონაცემებთან შედარებით.', learnMore: 'გადახრის მონიტორინგი სასიცოცხლოა საწარმოო სისტემებისთვის და ხშირად იწვევს გადაწვრთნას ან პაიპლაინის განახლებებს.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Precision and Recall', definition: 'Metrics that balance false positives and false negatives in classification tasks.', learnMore: 'Precision asks, "Of predicted positives, how many were correct?" Recall asks, "Of true positives, how many were found?"' },
    geo: { term: 'სიზუსტე და ამოხმობა', definition: 'მეტრიკები, რომლებიც კლასიფიკაციის ამოცანებში ცრუ დადებითებსა და ცრუ უარყოფითებს ამოწონებს.', learnMore: 'სიზუსტე კითხულობს: "პროგნოზირებული დადებითებიდან რამდენი იყო სწორი?" ამოხმობა კი: "ნამდვილი დადებითებიდან რამდენი მოიძებნა?"' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Regression', definition: 'A supervised learning task that predicts a continuous numeric value from input features.', learnMore: 'Linear regression models a straight-line relationship; polynomial and tree-based variants capture nonlinear patterns. Evaluation uses metrics like RMSE and R².' },
    geo: { term: 'რეგრესია', definition: 'ზედამხედველობითი სწავლების ამოცანა, რომელიც შეყვანის ნიშნებიდან უწყვეტ რიცხვით მნიშვნელობას პროგნოზირებს.', learnMore: 'წრფივი რეგრესია პირდაპირ ხაზობრივ კავშირს მოდელირებს; პოლინომიალური და ხეზე დაფუძნებული ვარიანტები არახაზობრივ პატერნებს იჭერს.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Classification', definition: 'A supervised learning task that assigns inputs to one of a fixed set of categories.', learnMore: 'Binary classification distinguishes two classes; multiclass handles more. Common algorithms include logistic regression, decision trees, and neural networks.' },
    geo: { term: 'კლასიფიკაცია', definition: 'ზედამხედველობითი სწავლების ამოცანა, რომელიც შეყვანებს კატეგორიების ფიქსირებულ ნაკრებს ანიჭებს.', learnMore: 'ბინარული კლასიფიკაცია ორ კლასს განასხვავებს; მრავალკლასიანი — მეტს. ჩვეულებრივი ალგორითმებია: ლოგისტიკური რეგრესია, გადაწყვეტილების ხეები და ნეირონული ქსელები.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Clustering', definition: 'An unsupervised learning task that groups similar data points without predefined labels.', learnMore: 'K-means, DBSCAN, and hierarchical clustering are common methods. Clusters reveal natural structure in data for segmentation and exploration.' },
    geo: { term: 'კლასტერიზაცია', definition: 'ზედამხედველობის გარეშე სწავლების ამოცანა, რომელიც მსგავს მონაცემთა წერტილებს ჯგუფებში აერთიანებს.', learnMore: 'K-means, DBSCAN და იერარქიული კლასტერიზაცია ჩვეულებრივი მეთოდებია. კლასტერები მონაცემებში ბუნებრივ სტრუქტურას ამჟღავნებს.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Decision Tree', definition: 'A model that makes predictions by splitting data through a series of branching conditions.', learnMore: 'Decision trees are interpretable but prone to overfitting. Pruning, depth limits, and ensemble methods like random forests address this weakness.' },
    geo: { term: 'გადაწყვეტილების ხე', definition: 'მოდელი, რომელიც პროგნოზებს გასაყოფი პირობების სერიით იღებს.', learnMore: 'გადაწყვეტილების ხეები ინტერპრეტირებადია, მაგრამ გადამეტებული მორგებისაგან, შემცირებისა და ანსამბლური მეთოდებით გამოსასწორებელია.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Random Forest', definition: 'An ensemble of decision trees trained on random data subsets whose predictions are averaged.', learnMore: 'Averaging many diverse trees reduces variance and improves generalization. Feature importance scores are a useful byproduct for model interpretation.' },
    geo: { term: 'შემთხვევითი ტყე', definition: 'გადაწყვეტილების ხეების ანსამბლი, რომელიც შემთხვევითი მონაცემთა ქვეჯგუფებზეა გაწვრთნილი და პროგნოზები საშუალოდდება.', learnMore: 'მრავალი მრავალფეროვანი ხის საშუალება ამცირებს დისპერსიას და გაზრდის განზოგადებას. ნიშნების მნიშვნელობის ქულები სასარგებლო დამატებითი პროდუქტია.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Support Vector Machine (SVM)', definition: 'A model that finds the widest possible margin boundary between classes in feature space.', learnMore: 'The kernel trick allows SVMs to handle nonlinear data by implicitly mapping it to higher dimensions. SVMs work well in high-dimensional, small-data settings.' },
    geo: { term: 'საყრდენი ვექტორების მანქანა (SVM)', definition: 'მოდელი, რომელიც ნიშნების სივრცეში კლასებს შორის ყველაზე ფართო შესაძლო ზღვარს პოულობს.', learnMore: 'ბირთვის ხრიკი SVMs-ს საშუალებას აძლევს გაუმკლავდეს არახაზობრივ მონაცემებს. SVMs კარგად მუშაობს მაღალგანზომილებიან, მცირე-მონაცემებიან გარემოში.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Cross-Validation', definition: 'A technique for estimating model performance by repeatedly splitting data into train and test folds.', learnMore: 'K-fold cross-validation rotates through k partitions so every sample is used for evaluation once. It gives a more reliable performance estimate than a single split.' },
    geo: { term: 'ჯვარედინი ვალიდაცია', definition: 'ტექნიკა მოდელის შესრულების შესასწავლად, მონაცემების სასწავლო და სასტესტო ნაწილებად განმეორებით დაყოფის გამოყენებით.', learnMore: 'K-fold ჯვარედინი ვალიდაცია k-ს პარტიციებს ერთდება, ყოველი ნიმუში შეფასებაში ერთხელ გამოიყენება. ეს უფრო სანდო შეფასებას იძლევა ერთ გაყოფასთან შედარებით.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Feature Engineering', definition: 'Transforming raw data into informative input representations that improve model performance.', learnMore: 'Techniques include normalization, encoding categoricals, creating interaction terms, and extracting domain-specific signals. Good features often matter more than model choice.' },
    geo: { term: 'ნიშნების ინჟინერია', definition: 'უხეში მონაცემების ინფორმაციულ შეყვანის წარმოდგენებად გარდაქმნა, რომელიც მოდელის მუშაობას აუმჯობესებს.', learnMore: 'ტექნიკებია: ნორმალიზაცია, კატეგორიული ცვლადების კოდირება, ინტერაქციური წევრების შექმნა. კარგი ნიშნები ხშირად მოდელის არჩევანზე მნიშვნელოვანია.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Loss Function', definition: 'A function measuring how far a model\'s predictions are from the true values during training.', learnMore: 'Common losses include mean squared error for regression and cross-entropy for classification. Minimizing the loss drives the optimization process.' },
    geo: { term: 'დანაკარგის ფუნქცია', definition: 'ფუნქცია, რომელიც ზომავს, რამდენად შორს არის მოდელის პროგნოზები სინამდვილის მნიშვნელობებისაგან გაწვრთნის დროს.', learnMore: 'ჩვეულებრივი დანაკარგებია: საშუალო კვადრატული შეცდომა რეგრესიისთვის და ჯვარედინი ენტროპია კლასიფიკაციისთვის.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Activation Function', definition: 'A mathematical function applied to each neuron\'s output to introduce non-linearity.', learnMore: 'ReLU is the most widely used activation in hidden layers. Sigmoid and softmax are common in output layers for binary and multiclass problems respectively.' },
    geo: { term: 'გააქტიურების ფუნქცია', definition: 'მათემატიკური ფუნქცია, რომელიც ყოველი ნეირონის გამომუშავებაზე გამოიყენება არახაზობრივობის შესატანად.', learnMore: 'ReLU ფარული შრეებში ყველაზე ფართოდ გამოყენებული გააქტიურებაა. Sigmoid და Softmax ჩვეულებრივია გამომავალ შრეებში.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Dropout', definition: 'A regularization technique that randomly deactivates neurons during training to prevent overfitting.', learnMore: 'Dropout forces the network to learn redundant representations. At inference time all neurons are active and outputs are scaled accordingly.' },
    geo: { term: 'დროფაუთი', definition: 'რეგულარიზაციის ტექნიკა, რომელიც გაწვრთნის დროს შემთხვევით გამორთავს ნეირონებს გადამეტებული მორგების თავიდან ასაცილებლად.', learnMore: 'დროფაუთი ქსელს ზედმეტი წარმოდგენების სწავლებას აიძულებს. ინფერენსის დროს ყველა ნეირონი აქტიურია.' },
  },
  {
    category: 'Machine Learning', level: 'Beginner',
    en: { term: 'Confusion Matrix', definition: 'A table comparing predicted class labels against actual labels across all categories.', learnMore: 'Each cell shows true positives, false positives, true negatives, and false negatives. Derived metrics include accuracy, precision, recall, and F1 score.' },
    geo: { term: 'შეცდომების მატრიცა', definition: 'ცხრილი, რომელიც პროგნოზირებულ კლასის ეტიკეტებს ნამდვილ ეტიკეტებს ადარებს ყველა კატეგორიაში.', learnMore: 'თითოეული უჯრა ნაჩვენებია: ნამდვილი დადებითები, ცრუ დადებითები, ნამდვილი უარყოფითები და ცრუ უარყოფითები.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'ROC Curve', definition: 'A plot of true positive rate vs. false positive rate as a classifier\'s threshold is varied.', learnMore: 'The area under the ROC curve (AUC) summarizes overall discrimination ability. AUC of 1.0 is perfect; 0.5 is no better than random guessing.' },
    geo: { term: 'ROC მრუდი', definition: 'ნამდვილი დადებითების მაჩვენებლის ჩვენება ცრუ დადებითების მაჩვენებელთან მიმართებაში კლასიფიკატორის ზღვრის ცვლილებისას.', learnMore: 'ROC მრუდის ქვეშ ფართობი (AUC) ზოგად გამარჩევ უნარს ასახავს. AUC 1.0 სრულყოფილია; 0.5 შემთხვევითი გამოცნობის ტოლია.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Principal Component Analysis (PCA)', definition: 'A linear technique that reduces data dimensionality by projecting onto directions of maximum variance.', learnMore: 'PCA removes correlated features and speeds up downstream models. The principal components are ordered so the first captures the most variance.' },
    geo: { term: 'მთავარ კომპონენტთა ანალიზი (PCA)', definition: 'წრფივი ტექნიკა, რომელიც მონაცემების განზომილებას ამცირებს მაქსიმალური დისპერსიის მიმართულებებზე პროექციის გზით.', learnMore: 'PCA ამოიღებს კორელირებულ ნიშნებს და უჩქარებს შემდგომ მოდელებს. კომპონენტები დალაგებულია ისე, რომ პირველი ყველაზე მეტ დისპერსიას იჭერს.' },
  },
  {
    category: 'Machine Learning', level: 'Advanced',
    en: { term: 'Batch Normalization', definition: 'A layer that normalizes activations within a mini-batch to stabilize and accelerate training.', learnMore: 'By reducing internal covariate shift, batch normalization allows higher learning rates and reduces sensitivity to initialization choices.' },
    geo: { term: 'ბატჩის ნორმალიზაცია', definition: 'შრე, რომელიც მინი-ბატჩის ფარგლებში გააქტიურებებს ნორმალიზაციას უკეთებს გაწვრთნის სტაბილიზაციისა და დაჩქარებისთვის.', learnMore: 'შიდა კოვარიატების გადახრის შემცირებით, ბატჩის ნორმალიზაცია უფრო მაღალ სწავლის სიჩქარეს იძლევა და ინიციალიზაციის მგრძნობიარობას ამცირებს.' },
  },
  {
    category: 'Machine Learning', level: 'Advanced',
    en: { term: 'Knowledge Distillation', definition: 'A compression technique where a smaller student model is trained to mimic a larger teacher model.', learnMore: 'The student learns from the teacher\'s soft probability outputs rather than hard labels, transferring generalization ability into a deployable, efficient model.' },
    geo: { term: 'ცოდნის დისტილაცია', definition: 'შეკუმშვის ტექნიკა, სადაც პატარა მოდელი (სტუდენტი) დიდ მოდელს (მასწავლებელს) ბაძავს.', learnMore: 'სტუდენტი სწავლობს მასწავლებლის რბილი ალბათობის გამომუშავებებიდან და არა მკაცრი ეტიკეტებიდან, განზოგადების უნარს ეფექტურ მოდელში გადასცემს.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Synthetic Data', definition: 'Artificially generated data used to supplement or replace real training examples.', learnMore: 'Synthetic data helps when real data is scarce, expensive, or privacy-sensitive. Quality control is critical — poor synthetic data can introduce new biases.' },
    geo: { term: 'სინთეზური მონაცემები', definition: 'ხელოვნურად გენერირებული მონაცემები, რომლებიც გამოიყენება რეალური სასწავლო მაგალითების შესავსებად ან ჩასანაცვლებლად.', learnMore: 'სინთეზური მონაცემები გვეხმარება, როდესაც რეალური მონაცემები მცირეა, ძვირია ან კონფიდენციალობისთვის მგრძნობიარეა.' },
  },
  {
    category: 'Machine Learning', level: 'Intermediate',
    en: { term: 'Ensemble Learning', definition: 'Combining predictions from multiple models to produce a more accurate and robust result.', learnMore: 'Bagging, boosting, and stacking are common ensemble strategies. Ensembles reduce variance and bias compared to any single constituent model.' },
    geo: { term: 'ანსამბლური სწავლება', definition: 'მრავალი მოდელის პროგნოზების გაერთიანება უფრო ზუსტი და მდგრადი შედეგის მისაღებად.', learnMore: 'Bagging, Boosting და Stacking ჩვეულებრივი ანსამბლური სტრატეგიებია. ანსამბლები ამცირებს დისპერსიასა და მიკერძოებას ნებისმიერ ცალკეულ მოდელთან შედარებით.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Generative Adversarial Network (GAN)', definition: 'A framework where a generator creates synthetic samples and a discriminator learns to distinguish them from real ones.', learnMore: 'The two networks improve through competition. GANs can produce photorealistic images but training can be unstable due to mode collapse and gradient issues.' },
    geo: { term: 'გენერაციულ-მოწინააღმდეგური ქსელი (GAN)', definition: 'ჩარჩო, სადაც გენერატორი სინთეზურ ნიმუშებს ქმნის, ხოლო დისკრიმინატორი სწავლობს მათ რეალურებისაგან განასხვავოს.', learnMore: 'ორი ქსელი კონკურენციის მეშვეობით უმჯობესდება. GAN-ებს შეუძლიათ ფოტორეალისტური სურათების შექმნა, თუმცა გაწვრთნა შეიძლება არასტაბილური იყოს.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Diffusion Model', definition: 'A generative model that learns to reverse a gradual noising process to synthesize new data.', learnMore: 'Models like DALL-E 3 and Stable Diffusion use diffusion. They produce high-quality, diverse outputs and have largely supplanted GANs for image generation.' },
    geo: { term: 'დიფუზიის მოდელი', definition: 'გენერაციული მოდელი, რომელიც თანდათანობითი ხმაურის დამატების პროცესის შებრუნებას სწავლობს ახალი მონაცემების სინთეზისთვის.', learnMore: 'DALL-E 3 და Stable Diffusion დიფუზიას იყენებს. ისინი მაღალი ხარისხის, მრავალფეროვან გამომუშავებებს ქმნის და გამოსახულების გენერაციაში GAN-ებს ჩაანაცვლა.' },
  },
  {
    category: 'Generative AI', level: 'Beginner',
    en: { term: 'Context Window', definition: 'The maximum amount of text a language model can process in a single input-output pass.', learnMore: 'Longer context windows let models handle full documents and long conversations. Context length is measured in tokens, not characters or words.' },
    geo: { term: 'კონტექსტის ფანჯარა', definition: 'ტექსტის მაქსიმალური რაოდენობა, რომელიც ენობრივ მოდელს შეუძლია ერთი შეყვანა-გამომუშავების პასაჟში დაამუშავოს.', learnMore: 'გრძელი კონტექსტის ფანჯრები მოდელებს სრული დოკუმენტების და გრძელი საუბრების დამუშავების საშუალებას აძლევს. კონტექსტის სიგრძე ტოკენებით იზომება.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Temperature', definition: 'A sampling parameter that controls randomness in a model\'s output token selection.', learnMore: 'Temperature near 0 makes outputs deterministic and focused; higher values increase creativity and diversity. It scales the logits before the softmax is applied.' },
    geo: { term: 'ტემპერატურა', definition: 'შერჩევის პარამეტრი, რომელიც მოდელის გამომუშავების ტოკენების შერჩევაში შემთხვევითობას აკონტროლებს.', learnMore: '0-თან ახლო ტემპერატურა გამომუშავებებს დეტერმინისტულს და ფოკუსირებულს ხდის; მაღალი მნიშვნელობები კრეატიულობასა და მრავალფეროვნებას ზრდის.' },
  },
  {
    category: 'Generative AI', level: 'Beginner',
    en: { term: 'Token', definition: 'The basic unit a language model reads and generates, typically a word fragment or character sequence.', learnMore: 'One token is roughly 4 characters in English. Context limits, pricing, and speed are all measured in tokens, making token efficiency a practical concern.' },
    geo: { term: 'ტოკენი', definition: 'ძირითადი ერთეული, რომელსაც ენობრივი მოდელი კითხულობს და გენერირებს, ჩვეულებრივ სიტყვის ფრაგმენტი ან სიმბოლოების თანმიმდევრობა.', learnMore: 'ერთი ტოკენი დაახლოებით 4 სიმბოლოა ინგლისურში. კონტექსტის ლიმიტები, ფასი და სიჩქარე ტოკენებით იზომება.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Few-Shot Learning', definition: 'Guiding a model to perform a task by providing a small number of input-output examples in the prompt.', learnMore: 'Few-shot prompting leverages in-context learning without any weight updates. More examples generally improve performance up to the context limit.' },
    geo: { term: 'მცირე-ნიმუშიანი სწავლება', definition: 'მოდელის დამისამართება ამოცანის შესასრულებლად პრომპტში მცირე რაოდენობის შეყვანა-გამომავლის მაგალითების მოწოდებით.', learnMore: 'მცირე-ნიმუშიანი პრომპტინგი კონტექსტში სწავლებას წონების განახლების გარეშე იყენებს. მეტი მაგალითი ზოგადად მუშაობას აუმჯობესებს.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Zero-Shot Learning', definition: 'Asking a model to perform a task it has not seen explicitly, relying solely on its pre-trained knowledge.', learnMore: 'Strong foundation models generalize well zero-shot because broad pre-training encodes diverse task knowledge. Clear task descriptions improve results.' },
    geo: { term: 'ნულ-ნიმუშიანი სწავლება', definition: 'მოდელისთვის ამოცანის შეთავაზება, რომელიც მან ცალსახად ვერ ნახა, მხოლოდ წინასწარ გაწვრთნილ ცოდნაზე დაყრდნობით.', learnMore: 'ძლიერი საბაზო მოდელები კარგად განაზოგადებს ნულ-ნიმუშიანად, რადგან ფართო წინასწარი გაწვრთნა მრავალფეროვან ამოცანის ცოდნას კოდირებს.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Chain-of-Thought Prompting', definition: 'A prompting strategy that asks the model to reason step-by-step before giving a final answer.', learnMore: 'Chain-of-thought significantly improves performance on math, logic, and multi-step reasoning tasks by externalizing the model\'s reasoning process.' },
    geo: { term: 'ფიქრის ჯაჭვის პრომპტინგი', definition: 'პრომპტინგის სტრატეგია, რომელიც მოდელს საბოლოო პასუხამდე ნაბიჯ-ნაბიჯ მსჯელობას სთხოვს.', learnMore: 'ფიქრის ჯაჭვი მნიშვნელოვნად აუმჯობესებს შედეგებს მათემატიკის, ლოგიკისა და მრავალსაფეხუριანი მსჯელობის ამოცანებში.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Reinforcement Learning from Human Feedback (RLHF)', definition: 'A training method that fine-tunes models using human preference rankings as a reward signal.', learnMore: 'RLHF is used to align LLMs like ChatGPT with human values. A reward model is trained on preference data and then used to optimize the language model via RL.' },
    geo: { term: 'ადამიანის გამოხმაურებით განმამტკიცებელი სწავლება (RLHF)', definition: 'გაწვრთნის მეთოდი, რომელიც მოდელებს ადამიანის პრეფერენციის რანგებს ჯილდოს სიგნალად იყენებს.', learnMore: 'RLHF გამოიყენება ChatGPT-ის მსგავსი LLM-ების ადამიანის ღირებულებებთან გასასწორებლად. ჯილდოს მოდელი გამოიყენება ენობრივი მოდელის RL-ით ოპტიმიზაციისთვის.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Multimodal Model', definition: 'A model that can process and generate across multiple types of data such as text, images, and audio.', learnMore: 'Multimodal models like GPT-4o combine vision and language encoders. They unlock applications like visual question answering and image captioning.' },
    geo: { term: 'მულტიმოდალური მოდელი', definition: 'მოდელი, რომელსაც შეუძლია ამუშავოს და გენერირება მოახდინოს მრავალი ტიპის მონაცემებზე, მაგ. ტექსტი, სურათები და აუდიო.', learnMore: 'GPT-4o-ს მსგავსი მულტიმოდალური მოდელები ხედვისა და ენის ენკოდერებს აერთიანებს. ისინი ვიზუალური კითხვა-პასუხის მსგავს აპლიკაციებს ხსნის.' },
  },
  {
    category: 'Generative AI', level: 'Beginner',
    en: { term: 'Foundation Model', definition: 'A large model trained on broad data that can be adapted to many downstream tasks.', learnMore: 'Foundation models are pre-trained once at great cost and then fine-tuned or prompted for specific uses, making them economically valuable across industries.' },
    geo: { term: 'საბაზო მოდელი', definition: 'დიდი მოდელი, გაწვრთნილი ფართო მონაცემებზე, რომლის მრავალ შემდგომ ამოცანასთან ადაპტაცია შეიძლება.', learnMore: 'საბაზო მოდელები ერთხელ ძვირადღირებული წინასწარი გაწვრთნის შემდეგ, კონკრეტული გამოყენებებისთვის ზუსტი მორგებით ან პრომპტინგით ადაპტირდება.' },
  },
  {
    category: 'Generative AI', level: 'Beginner',
    en: { term: 'Text-to-Image', definition: 'A generative AI capability that produces images from natural language descriptions.', learnMore: 'Models like Midjourney and Stable Diffusion map text embeddings to image space. Prompt quality, style words, and negative prompts strongly influence output.' },
    geo: { term: 'ტექსტიდან სურათი', definition: 'გენერაციული AI-ის შესაძლებლობა, რომელიც ბუნებრივი ენის აღწერებიდან სურათებს ქმნის.', learnMore: 'Midjourney-ისა და Stable Diffusion-ის მსგავსი მოდელები ტექსტის ვექტორებს სურათის სივრცეში გადასახავს. პრომპტის ხარისხი და სტილის სიტყვები გამომუშავებაზე ძლიერ მოქმედებს.' },
  },
  {
    category: 'Generative AI', level: 'Advanced',
    en: { term: 'Variational Autoencoder (VAE)', definition: 'A generative model that encodes inputs to a probabilistic latent space and decodes samples back to data.', learnMore: 'VAEs learn smooth, continuous latent representations useful for generation and interpolation. They underpin the latent spaces used in many diffusion pipelines.' },
    geo: { term: 'ვარიაციული ავტოენკოდერი (VAE)', definition: 'გენერაციული მოდელი, რომელიც შეყვანებს ალბათობრივ ლატენტურ სივრცეში კოდირებს და ნიმუშებს მონაცემებში უბრუნებს.', learnMore: 'VAE-ები ლატენტური სივრცის გლუვ, უწყვეტ წარმოდგენებს სწავლობს. ისინი მრავალი დიფუზიის პაიპლაინის ლატენტური სივრცეების საფუძველს ქმნის.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'In-Context Learning', definition: 'A model\'s ability to adapt its behavior based on examples or instructions provided in the prompt, without weight updates.', learnMore: 'In-context learning emerges from scale. The model identifies patterns from the prompt itself, enabling rapid task switching without any fine-tuning.' },
    geo: { term: 'კონტექსტში სწავლება', definition: 'მოდელის შესაძლებლობა მოახდინოს ქცევის ადაპტაცია პრომპტში მოწოდებული მაგალითების ან ინსტრუქციების საფუძველზე, წონების განახლების გარეშე.', learnMore: 'კონტექსტში სწავლება მასშტაბიდან იჩენს თავს. მოდელი თავად პრომპტიდან პატერნებს ამოიცნობს, რაც ზუსტი მორგების გარეშე სწრაფ ამოცანებს შორის გადართვას იძლევა.' },
  },
  {
    category: 'Generative AI', level: 'Intermediate',
    en: { term: 'Sampling Strategy', definition: 'The method used to select the next token during generation, such as greedy, top-k, or nucleus sampling.', learnMore: 'Top-p (nucleus) sampling draws from the smallest set of tokens whose cumulative probability exceeds p, balancing quality and diversity better than greedy decoding.' },
    geo: { term: 'შერჩევის სტრატეგია', definition: 'მეთოდი, გამოყენებული გენერაციისას შემდეგი ტოკენის შესარჩევად, მაგ. Greedy, Top-k ან Nucleus Sampling.', learnMore: 'Top-p (nucleus) sampling ტოკენების ყველაზე პატარა ნაკრებიდან იღებს, რომლის კუმულატიური ალბათობა p-ს აჭარბებს, რაც ხარისხსა და მრავალფეროვნებას ამოწონებს.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Language Model', definition: 'A probabilistic model that assigns likelihoods to sequences of words or tokens.', learnMore: 'Language models power autocomplete, speech recognition, and generation tasks. Modern neural LMs learn from vast text corpora using self-supervised objectives.' },
    geo: { term: 'ენობრივი მოდელი', definition: 'ალბათობრივი მოდელი, რომელიც სიტყვების ან ტოკენების თანმიმდევრობებს ალბათობებს ანიჭებს.', learnMore: 'ენობრივი მოდელები მართავს ავტოდასრულებას, მეტყველების ამოცნობასა და გენერაციის ამოცანებს. თანამედროვე ნეირონული LM-ები უზარმაზარი ტექსტის კორპუსიდან სწავლობს.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Machine Translation', definition: 'Automatically converting text from one human language to another using a model.', learnMore: 'Neural machine translation using encoder-decoder transformers has dramatically improved quality. Challenges include low-resource language pairs and cultural nuance.' },
    geo: { term: 'მანქანური თარგმანი', definition: 'ტექსტის ერთი ადამიანური ენიდან მეორეზე ავტომატური გარდაქმნა მოდელის გამოყენებით.', learnMore: 'ენკოდერ-დეკოდერ ტრანსფორმერების გამოყენებით ნეირონულმა მანქანურმა თარგმანმა ხარისხი მნიშვნელოვნად გააუმჯობესა.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Text Summarization', definition: 'Automatically condensing a long document into a shorter version retaining key information.', learnMore: 'Extractive summarization selects existing sentences; abstractive summarization generates new text. Evaluation uses ROUGE scores and human judgment.' },
    geo: { term: 'ტექსტის შეჯამება', definition: 'გრძელი დოკუმენტის ავტომატური შეკუმშვა მოკლე ვერსიად ძირითადი ინფორმაციის შენარჩუნებით.', learnMore: 'ექსტრაქტული შეჯამება არსებულ წინადადებებს ირჩევს; აბსტრაქტული ახალ ტექსტს გენერირებს. შეფასება ROUGE ქულებს იყენებს.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Question Answering', definition: 'An NLP task where a model produces a correct answer to a natural language question.', learnMore: 'Closed-book QA relies on model knowledge; open-book QA retrieves context first. Benchmarks like SQuAD and TriviaQA measure progress.' },
    geo: { term: 'კითხვა-პასუხი', definition: 'NLP-ამოცანა, სადაც მოდელი ბუნებრივი ენის კითხვაზე სწორ პასუხს წარმოქმნის.', learnMore: 'დახურული წიგნის QA მოდელის ცოდნაზეა დამოკიდებული; ღია წიგნის QA კონტექსტს ჯერ ეძებს. SQuAD-ისა და TriviaQA-ს მსგავსი ბენჩმარკები პროგრესსა ზომავს.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Perplexity', definition: 'A measure of how well a language model predicts a text sample; lower is better.', learnMore: 'Perplexity is the exponentiated average negative log-likelihood per token. It is used to compare language models on held-out data but does not capture all quality aspects.' },
    geo: { term: 'პერპლექსია', definition: 'საზომი, თუ რამდენად კარგად პროგნოზირებს ენობრივი მოდელი ტექსტის ნიმუშს; დაბალი უკეთესია.', learnMore: 'პერპლექსია არის ტოკენზე საშუალო უარყოფითი ლოგ-ალბათობის ექსპონენტი. გამოიყენება ენობრივი მოდელების შედარებისთვის, მაგრამ ყველა ხარისხის ასპექტს არ ფარავს.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Word2Vec', definition: 'A shallow neural model that learns dense word embeddings by predicting surrounding words in context.', learnMore: 'Word2Vec introduced the idea that word arithmetic reflects semantics (king − man + woman ≈ queen). It laid the groundwork for modern contextual embeddings.' },
    geo: { term: 'Word2Vec', definition: 'არაღრმა ნეირონული მოდელი, რომელიც კონტექსტში გარშემომყოფი სიტყვების პროგნოზირებით მკვრივ სიტყვის ვექტორებს სწავლობს.', learnMore: 'Word2Vec-მა შემოიტანა იდეა, რომ სიტყვების არითმეტიკა სემანტიკას ასახავს (მეფე − კაცი + ქალი ≈ დედოფალი). ჩაუყარა საფუძველი თანამედროვე კონტექსტური ვექტორებს.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Part-of-Speech Tagging', definition: 'Labeling each word in a sentence with its grammatical role such as noun, verb, or adjective.', learnMore: 'POS tags are a foundational NLP preprocessing step used in parsing, information extraction, and grammar checking pipelines.' },
    geo: { term: 'მეტყველების ნაწილების მარკირება', definition: 'წინადადების თითოეული სიტყვის სახელდება გრამატიკული როლის მიხედვით, მაგ. არსებითი, ზმნა ან ზედსართავი.', learnMore: 'მეტყველების ნაწილების მარკირება NLP-ის ძირეული წინასწარი დამუშავების ნაბიჯია, გამოყენებული სინტაქსური ანალიზის, ინფორმაციის ამოღებისა და გრამატიკის შემოწმების პაიპლაინებში.' },
  },
  {
    category: 'NLP', level: 'Intermediate',
    en: { term: 'Semantic Search', definition: 'Retrieving documents based on meaning rather than exact keyword matches.', learnMore: 'Semantic search encodes queries and documents as embeddings and finds nearest neighbors. It underlies RAG systems and modern enterprise search products.' },
    geo: { term: 'სემანტიკური ძიება', definition: 'დოკუმენტების მოძიება მნიშვნელობის საფუძველზე და არა ზუსტი საკვანძო სიტყვების შედარებით.', learnMore: 'სემანტიკური ძიება შეკითხვებსა და დოკუმენტებს ვექტორებად კოდირებს და ყველაზე ახლო მეზობლებს პოულობს. RAG სისტემებისა და თანამედროვე ძიების პროდუქტების საფუძველია.' },
  },
  {
    category: 'NLP', level: 'Advanced',
    en: { term: 'Coreference Resolution', definition: 'Identifying when different expressions in a text refer to the same real-world entity.', learnMore: 'Resolving pronouns and noun phrases to their antecedents is essential for reading comprehension, summarization, and dialogue systems.' },
    geo: { term: 'კო-რეფერენციის გარჩევა', definition: 'ტექსტში სხვადასხვა გამოთქმების იდენტიფიცირება, როდესაც ისინი ერთსა და იმავე რეალურ ობიექტს მიმართავს.', learnMore: 'ნაცვალსახელებისა და საარსებო ფრაზების მათ წინამორბედებთან დაკავშირება აუცილებელია კითხვის გაგებისთვის, შეჯამებისა და დიალოგის სისტემებისთვის.' },
  },
  {
    category: 'NLP', level: 'Beginner',
    en: { term: 'Text Classification', definition: 'Assigning predefined labels to text documents, such as topic, intent, or spam detection.', learnMore: 'Text classification is one of the most common NLP tasks. Fine-tuned transformer models now achieve near-human accuracy on many classification benchmarks.' },
    geo: { term: 'ტექსტის კლასიფიკაცია', definition: 'ტექსტური დოკუმენტებისთვის წინასწარ განსაზღვრული ეტიკეტების მინიჭება, მაგ. თემა, განზრახვა ან სპამის გამოვლენა.', learnMore: 'ტექსტის კლასიფიკაცია NLP-ის ყველაზე ჩვეულებრივი ამოცანებიდანაა. ზუსტად მორგებული ტრანსფორმერის მოდელები ადამიანთან მსგავს სიზუსტეს ბევრ ბენჩმარკზე აღწევს.' },
  },
  {
    category: 'Computer Vision', level: 'Beginner',
    en: { term: 'Image Classification', definition: 'Assigning a single category label to an entire image based on its visual content.', learnMore: 'ImageNet benchmarks drove major advances in image classification. CNNs and Vision Transformers (ViTs) now achieve superhuman accuracy on standard datasets.' },
    geo: { term: 'გამოსახულების კლასიფიკაცია', definition: 'მთელ გამოსახულებაზე ვიზუალური შინაარსის საფუძველზე ერთი კატეგორიის ეტიკეტის მინიჭება.', learnMore: 'ImageNet-ის ბენჩმარკებმა გამოსახულების კლასიფიკაციის ძირითადი პროგრესი განაპირობა. CNN-ები და ViT-ები ახლა ადამიანზე მაღალ სიზუსტეს სტანდარტულ მონაცემებზე აღწევს.' },
  },
  {
    category: 'Computer Vision', level: 'Beginner',
    en: { term: 'Data Augmentation', definition: 'Artificially expanding a training dataset by applying transformations like flips, crops, and color shifts.', learnMore: 'Augmentation reduces overfitting and improves robustness to real-world variation. Modern policies like RandAugment learn optimal augmentation strategies automatically.' },
    geo: { term: 'მონაცემთა გამდიდრება', definition: 'სასწავლო მონაცემების ხელოვნური გაფართოება ტრანსფორმაციების გამოყენებით, მაგ. გადატრიალება, მოჭრა და ფერის გადახრა.', learnMore: 'მონაცემთა გამდიდრება ამცირებს გადამეტებულ მორგებას და მდგრადობას ზრდის რეალური სამყაროს ცვლილებებთან. RandAugment-ის მსგავსი თანამედროვე მეთოდები ოპტიმალურ სტრატეგიებს ავტომატურად სწავლობს.' },
  },
  {
    category: 'Computer Vision', level: 'Intermediate',
    en: { term: 'Face Recognition', definition: 'Identifying or verifying a person\'s identity from facial features in an image or video.', learnMore: 'Modern systems embed faces into high-dimensional vectors and compare distances. Accuracy varies with lighting and angle; ethical concerns around surveillance are significant.' },
    geo: { term: 'სახის ამოცნობა', definition: 'პირის ვინაობის იდენტიფიცირება ან გადამოწმება სურათში ან ვიდეოში სახის ნიშნებიდან.', learnMore: 'თანამედროვე სისტემები სახეებს მაღალგანზომილებიან ვექტორებში კოდირებს და მანძილებს ადარებს. სიზუსტე განათებაზე და კუთხეზეა დამოკიდებული; სათვალთვალო სისტემებთან დაკავშირებული ეთიკური პრობლემები მნიშვნელოვანია.' },
  },
  {
    category: 'Computer Vision', level: 'Beginner',
    en: { term: 'Optical Character Recognition (OCR)', definition: 'Converting images of printed or handwritten text into machine-readable characters.', learnMore: 'Modern OCR combines detection and recognition stages. It powers document digitization, accessibility tools, and automated data extraction pipelines.' },
    geo: { term: 'ოპტიკური სიმბოლოების ამოცნობა (OCR)', definition: 'დაბეჭდილი ან ხელნაწერი ტექსტის გამოსახულებების მანქანით წასაკითხ სიმბოლოებად გარდაქმნა.', learnMore: 'თანამედროვე OCR გამოვლენისა და ამოცნობის ეტაპებს აერთიანებს. ამოქმედებს დოკუმენტების ციფრიზაციას, ხელმისაწვდომობის ხელსაწყოებს და ავტომატური მონაცემების ამოღების პაიპლაინებს.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Pose Estimation', definition: 'Detecting the positions of body joints to infer a person\'s posture and movement from images or video.', learnMore: 'Pose estimation is used in sports analytics, physical therapy, animation, and human-computer interaction. Models output skeletal keypoints with confidence scores.' },
    geo: { term: 'პოზის შეფასება', definition: 'სხეულის სახსრების პოზიციების გამოვლენა სურათებიდან ან ვიდეოდან ადამიანის პოზისა და მოძრაობის დასადგენად.', learnMore: 'გამოიყენება სპორტულ ანალიტიკაში, ფიზიოთერაპიაში, ანიმაციაში და ადამიანი-კომპიუტერის ინტერაქციაში. მოდელები ჩონჩხის საკვანძო წერტილებს სანდოობის ქულებთან ერთად გამოაქვს.' },
  },
  {
    category: 'Computer Vision', level: 'Intermediate',
    en: { term: 'Feature Extraction', definition: 'Transforming raw image pixels into compact, meaningful representations for downstream tasks.', learnMore: 'Pretrained CNN backbones like ResNet are commonly used as feature extractors. Their intermediate layers capture edges, textures, and semantic patterns at different scales.' },
    geo: { term: 'ნიშნების ამოღება', definition: 'უხეში გამოსახულების პიქსელების კომპაქტურ, მნიშვნელოვან წარმოდგენებად გარდაქმნა შემდგომი ამოცანებისთვის.', learnMore: 'ResNet-ის მსგავსი წინასწარ გაწვრთნილი CNN-ები ჩვეულებრივ ნიშნების ამომღებად გამოიყენება. მათი შუალედური შრეები კიდეებს, ტექსტურებსა და სემანტიკურ პატერნებს სხვადასხვა მასშტაბით იჭერს.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Depth Estimation', definition: 'Predicting the distance of objects from the camera using one or more images.', learnMore: 'Monocular depth estimation infers 3D structure from a single image using learned priors. Stereo and LiDAR-based methods provide ground truth for training.' },
    geo: { term: 'სიღრმის შეფასება', definition: 'ობიექტების კამერიდან მანძილის პროგნოზირება ერთი ან მეტი გამოსახულების გამოყენებით.', learnMore: 'მონოკულური სიღრმის შეფასება ერთი სურათიდან 3D სტრუქტურას სწავლის გზით იდენტიფიცირებს. სტერეო და LiDAR-ზე დაფუძნებული მეთოდები გაწვრთნისთვის ეტალონს გვაწვდის.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Video Understanding', definition: 'Analyzing temporal sequences of frames to recognize actions, events, and content in video.', learnMore: 'Video models must capture both spatial and temporal patterns. Applications include content moderation, sports analytics, and autonomous driving perception.' },
    geo: { term: 'ვიდეოს გაანალიზება', definition: 'კადრების დროითი თანმიმდევრობების ანალიზი ვიდეოში მოქმედებების, მოვლენებისა და შინაარსის ასამოცნობად.', learnMore: 'ვიდეოს მოდელებმა სივრცობრივი და დროითი პატერნები ორივე უნდა დაიჭიროს. გამოიყენება კონტენტის მოდერაციაში, სპორტულ ანალიტიკასა და ავტონომიური მართვის სისტემებში.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Instance Segmentation', definition: 'Detecting and delineating each individual object instance with a pixel-level mask.', learnMore: 'Unlike semantic segmentation, instance segmentation distinguishes separate objects of the same class. Models like Mask R-CNN and SAM are widely used.' },
    geo: { term: 'ინსტანციის სეგმენტაცია', definition: 'ყოველი ცალკეული ობიექტის ინსტანციის გამოვლენა და გამიჯვნა პიქსელის დონის ნიღბის გამოყენებით.', learnMore: 'სემანტიკური სეგმენტაციისგან განსხვავებით, ინსტანციის სეგმენტაცია ერთი კლასის ცალ-ცალკე ობიექტებს განასხვავებს. Mask R-CNN-ი და SAM ფართოდ გამოიყენება.' },
  },
  {
    category: 'Computer Vision', level: 'Beginner',
    en: { term: 'Bounding Box', definition: 'A rectangle defined by coordinates used to localize an object within an image.', learnMore: 'Bounding boxes are the standard annotation format for object detection. Intersection over Union (IoU) measures how well a predicted box overlaps with a ground-truth box.' },
    geo: { term: 'შემოსაზღვრელი ჩარჩო', definition: 'კოორდინატებით განსაზღვრული მართკუთხედი, რომელიც გამოიყენება ობიექტის სურათში ლოკალიზებისთვის.', learnMore: 'შემოსაზღვრელი ჩარჩოები ობიექტის გამოვლენის სტანდარტული ანოტაციის ფორმატია. IoU (Intersection over Union) ზომავს, რამდენად კარგად ემთხვევა პროგნოზირებული ჩარჩო ეტალონს.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Vision Transformer (ViT)', definition: 'A transformer architecture applied to images by treating fixed-size patches as token sequences.', learnMore: 'ViTs match or exceed CNNs on large datasets and scale better with compute. They have unified vision and language modeling under a single architecture.' },
    geo: { term: 'ხედვის ტრანსფორმერი (ViT)', definition: 'ტრანსფორმერის არქიტექტურა, გამოყენებული გამოსახულებებზე ფიქსირებული ზომის პაჩებს ტოკენების თანმიმდევრობად მოპყრობის გზით.', learnMore: 'ViT-ები დიდ მონაცემებზე CNN-ებს ადარდება ან სჯობია და გამოთვლასთან ერთად უკეთ მასშტაბირდება. ისინი ხედვისა და ენის მოდელირება ერთ არქიტექტურაში გააერთიანა.' },
  },
  {
    category: 'Computer Vision', level: 'Advanced',
    en: { term: 'Point Cloud', definition: 'A set of 3D coordinate points representing the surface geometry of objects in space.', learnMore: 'Point clouds are produced by LiDAR and depth cameras. Models like PointNet process them directly for 3D object detection and autonomous vehicle perception.' },
    geo: { term: 'წერტილთა ღრუბელი', definition: '3D კოორდინატების ნაკრები, რომელიც სივრცეში ობიექტების ზედაპირის გეომეტრიას წარმოადგენს.', learnMore: 'წერტილთა ღრუბლები LiDAR-ით და სიღრმის კამერებით იქმნება. PointNet-ის მსგავსი მოდელები მათ პირდაპირ ამუშავებს 3D ობიექტის გამოვლენისა და ავტონომიური მანქანის სენსორიკისთვის.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'AI Safety', definition: 'The field concerned with ensuring AI systems behave reliably and do not cause unintended harm.', learnMore: 'AI safety encompasses robustness, interpretability, and alignment research. It addresses both near-term deployment risks and longer-term concerns about advanced systems.' },
    geo: { term: 'AI-უსაფრთხოება', definition: 'სფერო, რომელიც AI სისტემების საიმედო ქცევის და განუზრახველი ზიანის თავიდან არიდების უზრუნველყოფით არის დაკავებული.', learnMore: 'AI-უსაფრთხოება მოიცავს მდგრადობის, ინტერპრეტირებადობის და გასწორების კვლევას. ეხება მოკლევადიანი განლაგების რისკებსა და გრძელვადიანი სისტემების გამოწვევებს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'Responsible AI', definition: 'A framework for developing and deploying AI in ways that are ethical, transparent, and accountable.', learnMore: 'Responsible AI practices include bias auditing, stakeholder consultation, clear documentation, and human oversight mechanisms throughout the model lifecycle.' },
    geo: { term: 'პასუხისმგებელი AI', definition: 'ჩარჩო AI-ის ეთიკური, გამჭვირვალე და ანგარიშვალდებული განვითარებისა და განლაგებისთვის.', learnMore: 'პასუხისმგებელი AI-ის პრაქტიკა მოიცავს მიკერძოების აუდიტს, დაინტერესებულ მხარეებთან კონსულტაციას, მკაფიო დოკუმენტაციას და ადამიანური ზედამხედველობის მექანიზმებს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'Transparency', definition: 'The principle that AI systems and their decision-making processes should be open and understandable.', learnMore: 'Transparency includes disclosing training data sources, model limitations, and evaluation results. It is a prerequisite for public trust and regulatory compliance.' },
    geo: { term: 'გამჭვირვალობა', definition: 'პრინციპი, რომ AI სისტემები და მათი გადაწყვეტილების მიღების პროცესები ღია და გასაგები უნდა იყოს.', learnMore: 'გამჭვირვალობა მოიცავს სასწავლო მონაცემების წყაროების, მოდელის შეზღუდვებისა და შეფასების შედეგების გამჟღავნებას. საჯარო ნდობის და მარეგულირებელ შესაბამისობის წინაპირობაა.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Accountability', definition: 'The obligation to explain and justify AI decisions and to accept responsibility for their consequences.', learnMore: 'Accountability requires clear ownership across design, deployment, and monitoring. Audit trails, incident logging, and governance structures support it in practice.' },
    geo: { term: 'ანგარიშვალდებულება', definition: 'ვალდებულება, ახსნა და გამართლება AI-ის გადაწყვეტილებები და მათი შედეგების პასუხისმგებლობის აღება.', learnMore: 'ანგარიშვალდებულება მოითხოვს მკაფიო საკუთრებას დიზაინის, განლაგების და მონიტორინგის მასშტაბით. ამ შემთხვევაში პრაქტიკაში მხარს უჭერს აუდიტის კვალი, ინციდენტების ჟურნალი და მმართველობის სტრუქტურები.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Model Card', definition: 'A structured document describing a model\'s intended uses, performance, and known limitations.', learnMore: 'Model cards standardize how models are communicated to developers and downstream users. They are increasingly required by regulators and enterprise procurement teams.' },
    geo: { term: 'მოდელის ბარათი', definition: 'სტრუქტურული დოკუმენტი, რომელიც აღწერს მოდელის განკუთვნილ გამოყენებებს, მუშაობასა და ცნობილ შეზღუდვებს.', learnMore: 'მოდელის ბარათები სტანდარტიზაციას ახდენს, თუ როგორ ეცნობება მოდელები დეველოპერებსა და შემდგომ მომხმარებლებს. სულ უფრო მეტად მოითხოვს მარეგულირებლები.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'AI Governance', definition: 'Policies, processes, and structures that guide the responsible development and use of AI in organizations.', learnMore: 'Governance frameworks define roles, review processes, and escalation paths. Regulatory initiatives like the EU AI Act are shaping mandatory governance requirements.' },
    geo: { term: 'AI-მმართველობა', definition: 'პოლიტიკა, პროცესები და სტრუქტურები, რომლებიც ორგანიზაციებში AI-ის პასუხისმგებელ განვითარებასა და გამოყენებას წარმართავს.', learnMore: 'მმართველობის ჩარჩოები განსაზღვრავს როლებს, განხილვის პროცესებსა და ესკალაციის გზებს. EU AI Act-ის მსგავსი მარეგულირებელი ინიციატივები სავალდებულო მმართველობის მოთხოვნებს აყალიბებს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Advanced',
    en: { term: 'Differential Privacy', definition: 'A mathematical framework that adds calibrated noise to data or outputs to limit what can be inferred about individuals.', learnMore: 'Differential privacy gives a formal privacy guarantee parameterized by epsilon. It is used in training data pipelines and model release to prevent membership inference attacks.' },
    geo: { term: 'დიფერენციალური კონფიდენციალობა', definition: 'მათემატიკური ჩარჩო, რომელიც მონაცემებს ან გამომუშავებებს გაწვრთნილ ხმაურს უმატებს ინდივიდების შესახებ დასკვნების შეზღუდვისთვის.', learnMore: 'დიფერენციალური კონფიდენციალობა ეფსილონით პარამეტრიზებულ ფორმალურ კონფიდენციალობის გარანტიას იძლევა. გამოიყენება სასწავლო მონაცემების პაიპლაინებში წევრობის ინფერენსის შეტევების თავიდან ასაცილებლად.' },
  },
  {
    category: 'Ethics & Safety', level: 'Advanced',
    en: { term: 'Federated Learning', definition: 'A training approach where model updates are computed locally on devices and only gradients are shared, not raw data.', learnMore: 'Federated learning preserves data privacy by keeping personal data on-device. It is used in mobile keyboard prediction and healthcare settings with sensitive records.' },
    geo: { term: 'ფედერაციული სწავლება', definition: 'გაწვრთნის მიდგომა, სადაც მოდელის განახლებები ლოკალურად გამოითვლება მოწყობილობებზე და მხოლოდ გრადიენტები გაიცემა, უხეში მონაცემების ნაცვლად.', learnMore: 'ფედერაციული სწავლება ინახავს მონაცემთა კონფიდენციალობას პირადი მონაცემების მოწყობილობაზე შენარჩუნებით. გამოიყენება მობილური კლავიატურის პროგნოზირებაში და ჯანდაცვაში.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Robustness', definition: 'A model\'s ability to maintain reliable performance under noisy inputs, distribution shifts, or adversarial conditions.', learnMore: 'Robustness testing includes evaluating on corrupted inputs, out-of-distribution data, and targeted attacks. It is a key criterion for safety-critical deployments.' },
    geo: { term: 'მდგრადობა', definition: 'მოდელის შესაძლებლობა, შეინარჩუნოს სანდო მუშაობა ხმაურიანი შეყვანების, განაწილების გადახრების ან მოწინააღმდეგური პირობების პირობებში.', learnMore: 'მდგრადობის ტესტირება მოიცავს გაფუჭებული შეყვანების, განაწილების გარე მონაცემებისა და მიზნობრივი შეტევების შეფასებას. უსაფრთხოება-კრიტიკული განლაგებებისთვის ძირითადი კრიტერიუმია.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Red Teaming', definition: 'Structured adversarial testing where experts attempt to elicit harmful or unintended model outputs.', learnMore: 'Red teaming probes for jailbreaks, biases, and safety failures before deployment. It complements automated evaluation with human creativity and domain expertise.' },
    geo: { term: 'წითელი გუნდი', definition: 'სტრუქტურული მოწინააღმდეგური ტესტირება, სადაც ექსპერტები ცდილობენ მოდელის მავნე ან განუზრახველი გამომუშავებების გამოწვევას.', learnMore: 'წითელი გუნდი განლაგებამდე jailbreak-ებს, მიკერძოებებს და უსაფრთხოების ხარვეზებს ამოწმებს. ავტომატური შეფასების ადამიანური კრეატიულობითა და სფეროს ექსპერტიზით შევსებაა.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'Deepfake', definition: 'Synthetic media in which a person\'s likeness is replaced or manipulated using generative AI.', learnMore: 'Deepfakes pose risks for misinformation, fraud, and non-consensual imagery. Detection tools and provenance standards like C2PA aim to counter their misuse.' },
    geo: { term: 'დიპფეიქი', definition: 'სინთეზური მედია, სადაც ადამიანის სახე გენერაციული AI-ის გამოყენებით ჩანაცვლებულია ან მანიპულირებულია.', learnMore: 'დიპფეიქები საფრთხეს უქმნის დეზინფორმაციას, თაღლითობასა და უხმო სურათებს. გამოვლენის ხელსაწყოები და C2PA-ს მსგავსი წარმოშობის სტანდარტები მათ ბოროტად გამოყენებასთან ბრძოლას ემსახურება.' },
  },
  {
    category: 'Ethics & Safety', level: 'Beginner',
    en: { term: 'Privacy', definition: 'The right of individuals to control how their personal information is collected and used by AI systems.', learnMore: 'AI privacy risks include training data memorization, re-identification from outputs, and inference of sensitive attributes. Regulations like GDPR mandate protective measures.' },
    geo: { term: 'კონფიდენციალობა', definition: 'ინდივიდების უფლება, გააკონტროლონ, თუ როგორ გროვდება და გამოიყენება მათი პირადი ინფორმაცია AI სისტემების მიერ.', learnMore: 'AI-ის კონფიდენციალობის რისკებია: სასწავლო მონაცემების დამახსოვრება, გამომუშავებებიდან ხელახალი იდენტიფიკაცია და სენსიტიური ატრიბუტების ინფერენსი. GDPR-ს მსგავსი რეგულაციები დამცავ ზომებს ავალებს.' },
  },
  {
    category: 'Ethics & Safety', level: 'Intermediate',
    en: { term: 'Model Audit', definition: 'A systematic evaluation of an AI model\'s behavior, fairness, and compliance against defined criteria.', learnMore: 'Audits may be internal or conducted by independent third parties. They examine training data, evaluation results, and real-world outcomes across demographic groups.' },
    geo: { term: 'მოდელის აუდიტი', definition: 'AI მოდელის ქცევის, სამართლიანობის და შესაბამისობის სისტემატური შეფასება განსაზღვრული კრიტერიუმების მიხედვით.', learnMore: 'აუდიტი შეიძლება იყოს შიდა ან დამოუკიდებელი მესამე მხარის ჩატარებული. ისინი იკვლევს სასწავლო მონაცემებს, შეფასების შედეგებს და რეალური სამყაროს შედეგებს დემოგრაფიულ ჯგუფებში.' },
  },
];

const rawCategories = ['All', ...new Set(terms.map((t) => t.category))];
const exploredTerms = new Set();

const cardsGrid = document.getElementById('cardsGrid');
const categoryFilters = document.getElementById('categoryFilters');
const searchInput = document.getElementById('searchInput');
const progressTracker = document.getElementById('progressTracker');
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
const randomBtn = document.getElementById('randomBtn');
const emptyState = document.getElementById('emptyState');
const heroEyebrow = document.getElementById('heroEyebrow');
const heroTitle = document.getElementById('heroTitle');
const heroSubtitle = document.getElementById('heroSubtitle');
const langBtns = document.querySelectorAll('.lang-btn');

let activeCategory = 'All';

function levelClass(level) {
  return level.toLowerCase();
}

function updateProgress(totalVisible = terms.length) {
  const t = ui[currentLang];
  progressTracker.textContent = `${exploredTerms.size} ${t.progressOf} ${totalVisible} ${t.progressLabel}`;
}

function updateStaticUI() {
  const t = ui[currentLang];
  heroEyebrow.textContent = t.eyebrow;
  heroTitle.textContent = t.title;
  heroSubtitle.textContent = t.subtitle;
  searchInput.placeholder = t.searchPlaceholder;
  randomBtn.textContent = t.randomBtn;
  emptyState.textContent = t.empty;

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  themeLabel.textContent = isDark ? t.themeLight : t.themeDark;
}

function createCard(item) {
  const lang = item[currentLang];
  const card = document.createElement('article');
  card.className = 'term-card';
  card.setAttribute('tabindex', '0');

  const catDisplay = categoryNames[currentLang][item.category] || item.category;
  const levelDisplay = levelNames[currentLang][item.level] || item.level;
  const learnMoreLabel = ui[currentLang].learnMoreLabel;

  card.innerHTML = `
    <div class="term-head">
      <h3 class="term-name">${lang.term}</h3>
      <span class="badge ${levelClass(item.level)}">${levelDisplay}</span>
    </div>
    <p class="category">${catDisplay}</p>
    <p class="definition">${lang.definition}</p>
    <p class="learn-more"><strong>${learnMoreLabel}:</strong> ${lang.learnMore}</p>
  `;

  const toggleExpand = () => {
    card.classList.toggle('expanded');
    exploredTerms.add(item.en.term);
    updateProgress();
  };

  card.addEventListener('click', toggleExpand);
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(); }
  });

  return card;
}

function renderCategories() {
  categoryFilters.innerHTML = '';
  rawCategories.forEach((cat) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-btn ${cat === activeCategory ? 'active' : ''}`;
    btn.textContent = categoryNames[currentLang][cat] || cat;
    btn.addEventListener('click', () => {
      activeCategory = cat;
      renderCategories();
      renderCards();
    });
    categoryFilters.appendChild(btn);
  });
}

function getFilteredTerms() {
  const query = searchInput.value.trim().toLowerCase();
  return terms.filter((item) => {
    const lang = item[currentLang];
    const inCategory = activeCategory === 'All' || item.category === activeCategory;
    const inSearch =
      lang.term.toLowerCase().includes(query) ||
      lang.definition.toLowerCase().includes(query) ||
      (categoryNames[currentLang][item.category] || item.category).toLowerCase().includes(query);
    return inCategory && inSearch;
  });
}

function renderCards() {
  const filtered = getFilteredTerms();
  cardsGrid.innerHTML = '';
  filtered.forEach((item) => cardsGrid.appendChild(createCard(item)));
  emptyState.classList.toggle('hidden', filtered.length > 0);
  updateProgress(filtered.length);
}

function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('theme', mode);
  const t = ui[currentLang];
  themeLabel.textContent = mode === 'dark' ? t.themeLight : t.themeDark;
}

function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) { setTheme(saved); return; }
  setTheme(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}

function highlightRandomCard() {
  const cards = Array.from(document.querySelectorAll('.term-card'));
  if (!cards.length) return;
  const randomCard = cards[Math.floor(Math.random() * cards.length)];
  randomCard.classList.add('highlight');
  randomCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (!randomCard.classList.contains('expanded')) randomCard.classList.add('expanded');
  const termName = randomCard.querySelector('.term-name')?.textContent;
  if (termName) {
    const match = terms.find((t) => t[currentLang].term === termName);
    if (match) exploredTerms.add(match.en.term);
    updateProgress(cards.length);
  }
  setTimeout(() => randomCard.classList.remove('highlight'), 1200);
}

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
  langBtns.forEach((btn) => btn.classList.toggle('active', btn.dataset.lang === lang));
  updateStaticUI();
  renderCategories();
  renderCards();
}

langBtns.forEach((btn) => {
  btn.addEventListener('click', () => setLang(btn.dataset.lang));
});

searchInput.addEventListener('input', renderCards);
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  setTheme(next);
});
randomBtn.addEventListener('click', highlightRandomCard);

const savedLang = localStorage.getItem('lang') || 'en';
initTheme();
setLang(savedLang);
