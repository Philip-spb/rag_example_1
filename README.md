# RAG Example 1

Проект для работы с приватными документами используя RAG (Retrieval-Augmented Generation), LangChain и LLM модели.

## Описание

Этот проект демонстрирует реализацию системы вопрос-ответ над документами с использованием:

- **RAG (Retrieval-Augmented Generation)** — поиск релевантной информации из документа перед отправкой в LLM
- **LangChain** — фреймворк для построения цепочек обработки с LLM
- **Chroma** — векторная база данных для хранения embeddings документов
- **HuggingFace Embeddings** — преобразование текста в векторные представления
- **OpenAI LLM** — генерация ответов на основе найденной информации
- **ConversationBufferMemory** — сохранение истории диалога между вопросами

## Функциональность

- ✅ Загрузка документов с интернета или локального хранилища
- ✅ Разбиение текста на chunks и создание embeddings
- ✅ Сохранение данных в Chroma (с персистентностью на диск)
- ✅ Поиск релевантных фрагментов документа по запросу
- ✅ Генерация ответов с сохранением контекста диалога
- ✅ Поддержка многошагового диалога с доступом к истории

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd rag_example_1
```

### 2. Установка зависимостей с помощью uv

```bash
uv sync
```

### 3. Активация виртуального окружения (опционально)

```bash
source .venv/bin/activate  # На Windows: .venv\Scripts\activate
```

## Конфигурация

### Переменные окружения

Создайте файл `.env` в корневой папке проекта с следующими переменными:

```bash
# API ключ OpenAI (обязателен)
OPENAI_API_KEY=sk-your-api-key-here

# Отключение параллелизма для tokenizers (рекомендуется)
TOKENIZERS_PARALLELISM=false
```

#### Получение OPENAI_API_KEY

1. Перейдите на https://platform.openai.com/account/api-keys
2. Создайте новый API ключ
3. Скопируйте его в `.env` файл

## Запуск проекта

### Базовое использование

```python
from main import DocumentProcessor

# Инициализация
processor = DocumentProcessor()

# Загрузка и обработка документа
processor.upload_document()
processor.split_embedding_and_storing_document()

# Создание цепочки для диалога
processor.make_conversational_retrieval_chain()

# Задание вопроса
response = processor.ask_question_llm("Ваш вопрос о документе")
print(response)
```

### Запуск через main.py

```bash
python main.py
```

## Структура проекта

```
rag_example_1/
├── main.py                  # Основной файл с классом DocumentProcessor
├── companyPolicies.txt      # Пример документа
├── chroma_data/             # Данные Chroma (создаётся при первом запуске)
├── .venv/                   # Виртуальное окружение (создаётся при uv sync)
├── .env                     # Переменные окружения (НЕ коммитить!)
├── .gitignore              # Git исключения
├── pyproject.toml           # Конфигурация проекта и зависимости
├── uv.lock                  # Lock файл зависимостей
└── README.md               # Этот файл
```

## Типичный рабочий процесс

1. **Инициализация**: `processor = DocumentProcessor()`
2. **Загрузка**: `processor.upload_document()` (загружает файл, если его нет)
3. **Индексация**: `processor.split_embedding_and_storing_document()` (создание embeddings и сохранение в Chroma)
4. **Подготовка цепочки**: `processor.make_conversational_retrieval_chain()` (инициализация LLM и памяти)
5. **Диалог**: `processor.ask_question_llm("вопрос")` (итеративное взаимодействие с сохранением истории)

## Как это работает

### RAG Pipeline

```
Вопрос пользователя
    ↓
Преобразование в embedding (HuggingFace)
    ↓
Поиск в Chroma (similarity search)
    ↓
Получение релевантных chunks документа
    ↓
Заполнение prompt'а найденными chunks + история диалога
    ↓
Отправка в OpenAI LLM
    ↓
Генерация ответа
    ↓
Сохранение в ConversationBufferMemory
    ↓
Вывод пользователю
```

## Важные параметры

- **chunk_size**: размер текстового chunk'а (по умолчанию 1000 символов)
- **chunk_overlap**: перекрытие между chunk'ами (по умолчанию 0)
- **memory_key**: название переменной в prompt'е для истории ("chat_history")
- **return_messages**: формат хранения истории (True — объекты Message, False — строка)

## Решение проблем

### Ошибка: "OPENAI_API_KEY not found"
Проверьте, что `.env` файл находится в корне проекта и содержит валидный API ключ.

### Ошибка: "TOKENIZERS_PARALLELISM conflicts with OpenMP threads"
Добавьте `TOKENIZERS_PARALLELISM=false` в `.env` файл.

### Модель не видит предыдущие вопросы
Убедитесь, что:
- `memory` передана в `ConversationalRetrievalChain`
- `return_messages=False` для простоты, или правильно сериализуются Message объекты в строку

### Chroma данные теряются при перезапуске
Убедитесь, что используется `persist_directory="./chroma_data"` при создании Chroma.

## Требования

- Python 3.8+
- OpenAI API ключ
- Интернет-соединение для загрузки embeddings моделей

## Лицензия

MIT
