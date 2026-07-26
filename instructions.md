python -m venv env

pip install -r requirements.txt

pip install fastapi uvicorn pydantic joblib pandas scikit-learn

npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install lucide-react



run below codes
uvicorn main:app --reload

cd frontend
npm run dev

story.md
Orders fly in, the warehouse clears out, and revenue hits an all-time high. It feels like a massive win. But when the dust settles, net profit hasn't moved an inch.

The common trap? Confusing volume with value. We lower prices to keep momentum high, mistakenly trading our margins for temporary sales spikes.

For week 9, I stripped this problem down to first principles. Using historical Indian e-commerce transaction logs to ground the model in real-world market dynamics, I built a system that stops chasing volume and starts optimizing for profit. Instead of asking, 'How do we sell the most items?', the model asks, 'What is the exact balance between volume and margin?' It identifies the precise discount that keeps products moving while maximizing total net profit.

Great pricing shouldn't just clear a warehouse; it should strengthen the bottom line.

Tech Stack: Python, XGBoost, Random Forest, FastAPI, React.js, Tailwind CSS, Render, Vercel

You can check out the live dashboard here [https://prescriptive-pricing-architecture.vercel.app]


Problem: Trading margin for volume in e-commerce pricing.

Week 9: Built an ML tool using XGBoost and Random Forest to optimize for profit instead.

Stack: Python, FastAPI, React, Tailwind.