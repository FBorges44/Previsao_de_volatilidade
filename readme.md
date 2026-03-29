# 📉 Previsão de volatilidade do Mercado

Este projeto foi criado para responder a uma pergunta simples, mas valiosa: **O quão agitado (volátil) o mercado de Bitcoin estará amanhã?** Saber se o mercado vai oscilar muito ou pouco é fundamental para investidores e empresas protegerem seu dinheiro.

---

## 💡 O que este projeto faz?
Eu construí um sistema que analisa o passado do Bitcoin e o que as notícias estão dizendo hoje para prever a movimentação de amanhã. O sistema compara duas formas de pensar:
1. **A Matemática Clássica (GARCH):** Regras matemáticas tradicionais usadas por bancos há décadas.
2. **Inteligência Artificial (LSTM):** Uma rede neural que "aprende" com o tempo, simulando a memória humana para identificar padrões complexos.

---

## 🚀 Diferenciais do Projeto
* **Lê Notícias Automaticamente:** O sistema usa uma técnica chamada **NLP (Processamento de Linguagem Natural)** para ler notícias financeiras e classificar se o "humor" do mercado é de otimismo ou medo.
* **Dados em Tempo Real:** Conexão direta com a corretora (Binance) para pegar os preços mais recentes.
* **Comparação de Resultados:** O projeto mostra visualmente qual método acertou mais.

---

## 🛠️ Ferramentas que usei
* **Python:** A base de tudo (linguagem de programação).
* **Deep Learning:** Para criar a "memória" da nossa IA.
* **FinBERT:** Uma IA especializada em entender textos do mundo das finanças.
* **Pandas:** Para organizar as tabelas de preços como se fossem um "Excel super potente".

---

## 📊 O que descobri?
Nos meus testes, a **Inteligência Artificial (IA)** se saiu melhor que a matemática comum, especialmente quando o mercado está muito instável. Isso acontece porque a IA consegue entender o impacto emocional das notícias, algo que fórmulas puras nem sempre captam.

---

## 📂 Como o projeto está organizado?
- `data/`: Onde guardamos os preços do Bitcoin.
- `models/`: O "cérebro" do sistema (os códigos da IA).
- `nlp/`: A parte que lê e entende as notícias.
- `main.py`: O botão de "Play" para rodar tudo.

---

## 👨‍💻 Sobre mim
**Francisco**
* Estudante de Computação no **IFPI** e Residente no **EmbarcaTech**.
* Apaixonado por transformar dados brutos em decisões inteligentes.
* Desenvolvedor focado em Python, Machine Learning e Inovação.
