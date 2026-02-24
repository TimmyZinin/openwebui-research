# Lisa Solovyova — Промпты для генерации реалистичных фото

> Модель: Gemini 2.5 Flash Image ("Nano Banana") — `gemini-2.5-flash-image`
> Или: Google AI Studio → Gemini → "Generate image"
> Цель: максимальный фотореализм, НЕ аватарный стиль

## Якорная строка (добавлять в КАЖДЫЙ промпт)

```
A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She has a warm genuine smile, light natural makeup, and a confident yet approachable demeanor. She looks like a real person photographed with a professional camera.
```

## 5 Hero Shots

### 1. Professional Portrait (для HeyGen Photo Avatar)
```
Ultra-realistic professional headshot photograph of a real person. A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She has a warm genuine smile, light natural makeup, and a confident yet approachable demeanor. She is wearing a cream cashmere turtleneck sweater in a modern coworking space. Soft natural window light from the left. Shot on Canon EOS R5, 85mm f/1.4 lens, shallow depth of field. Photorealistic, 8K quality, indistinguishable from a real photograph. No illustration, no cartoon, no anime, no painting.
```

### 2. Cafe Casual
```
Ultra-realistic candid photograph of a real person. A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She is sitting in a cozy European-style café, holding a latte cup, looking slightly to the right with a warm smile. Wearing a light sage-green blouse. Warm ambient lighting. Shot on Sony A7IV, 50mm f/1.8, natural bokeh background. Photorealistic, indistinguishable from a real photo.
```

### 3. Speaker / Presentation
```
Ultra-realistic event photograph. A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She is standing on a small stage, gesturing with one hand while speaking to an audience. Wearing a fitted navy blazer over a white top. Conference-style lighting. Shot on Nikon Z8, 70mm. Photorealistic, editorial quality, real person.
```

### 4. Home Office
```
Ultra-realistic lifestyle photograph. A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She is sitting at a clean Scandinavian-style desk with a MacBook, looking at the camera with a friendly expression. Wearing a soft beige knit sweater. Morning golden hour light through sheer curtains. Shot on Fujifilm X-T5, 35mm f/1.4. Instagram-quality, photorealistic.
```

### 5. Outdoor Park
```
Ultra-realistic outdoor portrait photograph. A 28-year-old Russian woman named Liza. She has warm honey-brown hair, slightly wavy, shoulder-length, parted to the left. Her face is oval-shaped with soft jawline, natural cheekbones, warm hazel-green eyes with golden flecks, light freckles across the nose bridge. She is walking in a green city park, wearing a light denim jacket over a white t-shirt. Natural sunlight, slight wind in her hair. Genuine, relaxed expression. Shot on Canon R6 Mark II, 85mm f/1.2. Street photography style, photorealistic, real person.
```

## Как генерировать

### Вариант A: Google AI Studio (вручную)
1. Зайти на https://aistudio.google.com/
2. Выбрать модель Gemini 2.5 Flash (Image)
3. Вставить промпт
4. Скачать результат

### Вариант B: API (когда квота восстановится)
```bash
curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"PROMPT_HERE"}]}],"generationConfig":{"responseModalities":["TEXT","IMAGE"]}}'
```

## Для HeyGen Photo Avatar
- Нужно 10-15 фото с РАЗНЫХ ракурсов и в разной одежде
- Лицо должно быть чётко видно, глаза открыты
- Разное освещение (но всегда видно лицо)
- Рекомендуется: 3:4 или 1:1 aspect ratio для портретов
