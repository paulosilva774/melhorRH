import streamlit as st
import os
import asyncio # Importa asyncio para rodar funções assíncronas
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from datetime import date, datetime
import textwrap
# import requests # Não usado, pode remover
import warnings
import re
import pandas as pd

warnings.filterwarnings("ignore")

# --- Configuração da API Key ---
# Use st.secrets para obter a chave API de forma segura no Streamlit
# No Streamlit Cloud, adicione [secrets] GOOGLE_API_KEY="SUA_CHAVE_AQUI"
# em um arquivo chamado .streamlit/secrets.toml
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("API Key do Google não encontrada. Por favor, configure GOOGLE_API_KEY nos segredos do Streamlit.")
    st.stop() # Para a execução se a chave não estiver configurada

# Configura o cliente da SDK do Gemini
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Erro ao inicializar o cliente da API Google GenAI: {e}")
    st.stop()

# Define os modelos a serem usados (conforme o arquivo anexo)
# Mantendo os modelos especificados, mas Flash é geralmente mais rápido e barato
MODELO_RAPIDO = "gemini-1.5-flash-latest" # Versão mais recente do Flash
MODELO_ROBUSTO = "gemini-1.5-pro-latest"  # Versão Pro como "robusto"

# Cria um serviço de sessão em memória
# Inicializado fora das funções para ser persistente na execução (Embora em Streamlit,
# a persistência pode exigir st.session_state dependendo de como é usado)
# Para este caso simples, inicializar aqui é suficiente para a execução única no run_async
session_service = InMemorySessionService()

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
# (Adaptada para Streamlit, removendo displays IPython)
async def call_agent(agent: Agent, message_text: str) -> str:
    # Usa um ID de sessão baseado no agente para clareza, ou poderia ser fixo para um usuário único
    session_id = f"{agent.name}_session"
    user_id = "streamlit_user" # ID de usuário fixo para a sessão do Streamlit

    try:
        # Tenta criar a sessão. Se já existir (ex: em reruns), pega a existente.
        # InMemorySessionService.create_session pode falhar se já existir.
        # Uma alternativa mais robusta para Streamlit seria usar st.session_state para gerenciar sessões ADK.
        # Para simplicidade, vamos tentar pegar a existente ou criar.
        # ADK session management in Streamlit requires careful handling due to reruns.
        # A simple approach for single execution: always create fresh or handle error.
        # Let's stick to simple create for now, assuming a fresh run per button click logic.
        session = await session_service.create_session(app_name=agent.name, user_id=user_id)
    except Exception as e:
         # If create fails, try getting existing, or simply proceed if create is idempotent enough
         # (InMemorySessionService create is not idempotent, it raises ValueError)
         # A robust ADK+Streamlit integration needs more sophisticated session management
         # For demo purposes, we proceed assuming create works or can be adapted.
         # Let's re-evaluate: ADK sessions are per user/app. For a single user in Streamlit,
         # maybe a single session per agent is okay for a single execution flow.
         # Let's just create it, assuming the runner handles subsequent messages if needed.
         # The simplest path: create session every time for a fresh start per call.
         # This might not fully utilize session history if that was intended.
         # Let's assume this simple approach is okay for this use case.
         session = await session_service.create_session(app_name=agent.name, user_id=user_id)


    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Use st.empty() or a dedicated area to show progress if needed
    # For simplicity, we just collect the final response here
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              # Não adicione quebra de linha extra se já terminar com uma
              if not final_response.endswith('\n'):
                  final_response += "\n"

    # Limpar a sessão após o uso, se desejado para evitar acúmulo em InMemorySessionService
    # await session_service.delete_session(user_id=user_id, session_id=session.id)
    # Nota: Deletar a sessão pode ser problemático se a mesma sessão ID for usada por múltiplas chamadas
    # O ideal é que cada sequência de agente use um session_id único ou gerencie-o via st.session_state

    return final_response

# Função auxiliar para formatar texto em Markdown (retorna string)
def to_markdown_string(text):
  # ADK sometimes returns bullet points as '•'. Convert them to standard markdown '*'
  text = text.replace('•', '*')
  # Optional: Indent blocks if needed, but st.markdown usually handles code blocks well
  # return textwrap.indent(text, '> ', predicate=lambda _: True) # Removed indentation for cleaner look in Streamlit
  return text

##########################################
# --- Agente 1: Analisador de Nascimento --- #
##########################################
async def agente_analisador(data_nascimento):
    # Use st.spinner para mostrar que algo está acontecendo
    with st.spinner("Executando Agente 1: Analisador de Nascimento..."):
        analisador = Agent(
            name="agente_analisador",
            model=MODELO_RAPIDO,
            instruction="""
            Você é um analista de personalidade e propósito de vida com base na data de nascimento.
            Sua tarefa é fornecer análises profundas e precisas sobre a personalidade, padrões emocionais,
            caminhos de carreira e desafios pessoais com base na data de nascimento fornecida.
            Use a ferramenta de busca do Google (google_search) para obter informações relevantes e
            garantir que as análises sejam fundamentadas e úteis.
            Formate a saída usando Markdown, com títulos para cada seção (1 a 6).
            """,
            description="Agente que analisa a personalidade e o propósito de vida com base na data de nascimento",
            tools=[google_search]
        )

        entrada_do_agente_analisador = f"""
        Data de Nascimento: {data_nascimento}

        Realize as seguintes análises, formatando cada resposta com um título Markdown (# ou ##):

        1. **Decodificador de Personalidade pela Data de Nascimento:** Com base na data de nascimento {data_nascimento}, descreva meus pontos fortes naturais, padrões emocionais e como me comporto em relacionamentos — que seja profundo, específico e psicologicamente preciso.
        2. **Roteiro da Infância:** Usando a data de nascimento {data_nascimento}, escreva um perfil psicológico de como minha infância moldou minha personalidade, hábitos e tomada de decisões hoje — seja gentil, mas revelador.
        3. **Analisador de Propósito Profissional:** Dada a data de nascimento {data_nascimento}, quais caminhos de carreira combinam com meus traços de personalidade, valores e talentos naturais? Sugira áreas, funções e ambientes de trabalho.
        4. **Detector de Auto-Sabotagem:** Com base na data {data_nascimento}, quais são meus hábitos de auto-sabotagem mais prováveis e como eles aparecem no dia a dia? Dê soluções práticas com base na psicologia.
        5. **Mapa de Gatilhos Emocionais:** Usando a data de nascimento {data_nascimento}, explique o que geralmente me desencadeia emocionalmente, como eu costumo reagir e como posso desenvolver resiliência emocional em torno desses padrões.
        6. **Escaneamento de Energia nos Relacionamentos:** Com base na data de nascimento {data_nascimento}, descreva como eu dou e recebo amor, o que preciso de um parceiro e que tipo de pessoa eu naturalmente atraio.
        """

        analises = await call_agent(analisador, entrada_do_agente_analisador)
        return analises

################################################
# --- Agente 2: Identificador de Melhorias --- #
################################################
async def agente_melhorias(data_nascimento, analises_agente1):
     with st.spinner("Executando Agente 2: Identificador de Melhorias..."):
        melhorias = Agent(
            name="agente_melhorias",
            model=MODELO_RAPIDO,
            instruction="""
            Você é um consultor de desenvolvimento pessoal. Sua tarefa é analisar as análises fornecidas
            anteriormente e identificar áreas de melhoria em cada uma das seis
            categorias (Personalidade, Infância, Propósito Profissional, Auto-Sabotagem, Gatilhos Emocionais, Relacionamentos).
            Seja específico e forneça sugestões práticas para o desenvolvimento pessoal para cada área.
            Formate a saída usando Markdown, com títulos para cada área de melhoria.
            """,
            description="Agente que identifica pontos de melhoria nas análises do Agente 1",
            # tools=[google_search] # Pode ser útil para buscar técnicas de melhoria
        )

        entrada_do_agente_melhorias = f"""
        Data de Nascimento: {data_nascimento}
        Análises do Agente 1:
        ---
        {analises_agente1}
        ---

        Com base nas análises acima, para cada uma das seis áreas (Personalidade, Infância, Propósito Profissional, Auto-Sabotagem, Gatilhos Emocionais, Relacionamentos), identifique áreas de melhoria e
        forneça sugestões práticas para o desenvolvimento pessoal. Formate cada seção com um título Markdown (# ou ##).
        """

        pontos_de_melhoria = await call_agent(melhorias, entrada_do_agente_melhorias)
        return pontos_de_melhoria

######################################
# --- Agente 3: Buscador de Pessoas de Sucesso --- #
######################################
# Função adaptada para retornar um DataFrame pandas
async def agente_buscador_sucesso(data_nascimento):
    with st.spinner("Executando Agente 3: Buscador de Pessoas de Sucesso..."):
        buscador_sucesso = Agent(
            name="agente_buscador_sucesso",
            model=MODELO_ROBUSTO, # Usando modelo mais robusto para busca
            instruction="""
                Você é um pesquisador de pessoas de sucesso brasileiras. Sua tarefa é buscar na internet 5 homens e 5 mulheres
                que nasceram na data fornecida e que alcançaram sucesso em suas áreas de atuação, e que sejam brasileiros.
                Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" e a data completa (dia, mês, ano)
                para garantir que os resultados sejam apenas de pessoas do Brasil nascidas nessa data.
                Use a ferramenta de busca do Google (google_search) para encontrar as informações e o site de onde tirou a informação.
                **Formate sua resposta como uma lista Markdown, onde cada item representa uma pessoa e inclui Nome, Profissão, No que tem sucesso e Site da Informação.**
                Exemplo:
                * Nome: [Nome da Pessoa] | Profissão: [Profissão] | Sucesso: [Descrição do Sucesso] | Site: [URL da Fonte]
                Repita este formato para 5 homens e 5 mulheres.
                """,
            description="Agente que busca pessoas de sucesso brasileiras nascidas na mesma data",
            tools=[google_search]
        )

        entrada_do_agente_buscador_sucesso = f"""
        Busque na internet 5 homens e 5 mulheres que nasceram na data {data_nascimento} e que alcançaram sucesso
        em suas áreas de atuação, e que sejam brasileiros. Formate a saída como uma lista Markdown
        usando o formato: "* Nome: [Nome] | Profissão: [Profissão] | Sucesso: [Descrição] | Site: [URL]"
        """

        tabela_markdown_str = await call_agent(buscador_sucesso, entrada_do_agente_buscador_sucesso)

        # --- Parsing da string Markdown para DataFrame ---
        data = []
        # A regex busca por linhas que começam com '*' seguido de espaço, e então captura os campos.
        # Adapte a regex se o formato exato de saída do modelo variar.
        pattern = re.compile(r"^\*\s*Nome:\s*(.*?)\s*\|\s*Profissão:\s*(.*?)\s*\|\s*Sucesso:\s*(.*?)\s*\|\s*Site:\s*(.*?)\s*$", re.MULTILINE)

        for match in pattern.finditer(tabela_markdown_str):
            nome, profissao, sucesso, site = match.groups()
            data.append([nome.strip(), profissao.strip(), sucesso.strip(), site.strip()])

        df = pd.DataFrame(data, columns=["Nome", "Profissão", "Sucesso", "Site da Informação"])

        return df # Retorna o DataFrame

##########################################
# --- Agente 4: Gerador de Relatório Final --- #
##########################################
async def agente_relatorio_final(data_nascimento, analises, melhorias, tabela_sucesso_df):
    with st.spinner("Executando Agente 4: Gerador de Relatório Final..."):
        # Converte o DataFrame da tabela de sucesso para uma string Markdown para incluir no prompt do Agente 4
        # Use to_markdown para um formato legível pelo LLM
        tabela_sucesso_md = tabela_sucesso_df.to_markdown(index=False)


        relatorio = Agent(
            name="agente_relatorio",
            model=MODELO_RAPIDO,
            instruction="""
            Você é um gerador de relatórios finais de análise de personalidade com base na data de nascimento.
            Sua tarefa é combinar as análises fornecidas, os pontos de melhoria e a lista de pessoas de sucesso
            em um relatório final coerente, otimista e motivador.
            Estruture o relatório com títulos claros em Markdown (#, ##).
            Comece com uma introdução sobre a análise da data de nascimento.
            Inclua as seções de Análises de Personalidade e Pontos de Melhoria.
            Apresente a lista de Pessoas de Sucesso nascidas na mesma data como inspiração, mencionando que a tabela está anexa ou incluída (copie o conteúdo da tabela fornecida).
            Conclua o relatório com uma mensagem de incentivo e empoderamento.
            Use um tom positivo e encorajador em todo o relatório.
            """,
            description="Agente que gera o relatório final combinando todas as análises",
            # tools=[google_search] # Removida ferramenta de busca
        )

        entrada_do_agente_relatorio = f"""
        Data de Nascimento Analisada: {data_nascimento}

        Conteúdo das Análises de Personalidade:
        ---
        {analises}
        ---

        Conteúdo dos Pontos de Melhoria:
        ---
        {melhorias}
        ---

        Lista de Pessoas de Sucesso Nascidas na Mesma Data (formato tabela/lista):
        ---
        {tabela_sucesso_md}
        ---

        Combine as informações acima em um relatório final otimista e motivador usando Markdown.
        Inclua todos os detalhes relevantes das seções anteriores.
        Apresente a lista de pessoas de sucesso de forma clara.
        Conclua com uma mensagem de incentivo.
        """

        relatorio_final = await call_agent(relatorio, entrada_do_agente_relatorio)
        return relatorio_final

##########################################
# --- Aplicação Streamlit --- #
##########################################

st.title("🌟 Analisador de Personalidade e Propósito de Vida 🌟")
st.markdown("Descubra insights sobre sua personalidade, desafios, caminhos de carreira e inspirações com base na sua data de nascimento.")

# Input da data de nascimento
data_nascimento_str = st.text_input(
    "Por favor, digite sua **DATA DE NASCIMENTO** no formato DD/MM/AAAA:",
    key="birth_date_input"
)

# Botão para iniciar a análise
run_button = st.button("✨ Gerar Relatório ✨")

# Container para exibir o relatório final e o botão de download
report_container = st.empty() # Placeholder para o relatório dinâmico

# Lógica de execução quando o botão é clicado
if run_button:
    if not data_nascimento_str:
        st.warning("Por favor, digite sua data de nascimento.")
    else:
        try:
            # Validar o formato da data
            data_objeto = datetime.strptime(data_nascimento_str, '%d/%m/%Y')
            st.info(f"Analisando a data de nascimento: {data_nascimento_str}")

            # Limpa resultados anteriores no state
            if 'final_report_md' in st.session_state:
                 del st.session_state['final_report_md']
            if 'sucesso_df' in st.session_state:
                 del st.session_state['sucesso_df']

            # --- Execução Sequencial dos Agentes (Envelopada em asyncio.run) ---
            # Criamos uma função async para orquestrar as chamadas assíncronas dos agentes
            async def run_all_agents(dob_str):
                # Não exibe resultados intermediários explicitamente no UI principal para manter a limpeza
                # Mas você pode adicionar st.write/st.markdown aqui se quiser ver cada etapa
                analises_agente1_result = await agente_analisador(dob_str)
                # st.markdown("### Resultado Agente 1:") # Opcional: Exibir resultados intermediários
                # st.markdown(to_markdown_string(analises_agente1_result))

                pontos_de_melhoria_result = await agente_melhorias(dob_str, analises_agente1_result)
                # st.markdown("### Resultado Agente 2:")
                # st.markdown(to_markdown_string(pontos_de_melhoria_result))

                tabela_sucesso_df_result = await agente_buscador_sucesso(dob_str)
                # st.markdown("### Resultado Agente 3 - Pessoas de Sucesso:")
                # st.dataframe(tabela_sucesso_df_result) # Exibe o DataFrame intermediário

                # Armazena o DataFrame no session_state para uso posterior (ex: exibir novamente)
                st.session_state['sucesso_df'] = tabela_sucesso_df_result

                relatorio_final_result = await agente_relatorio_final(
                    dob_str,
                    analises_agente1_result,
                    pontos_de_melhoria_result,
                    tabela_sucesso_df_result # Passa o DataFrame
                )
                return relatorio_final_result

            # Executa a função assíncrona que chama todos os agentes
            final_report_content = asyncio.run(run_all_agents(data_nascimento_str))

            # Converte o relatório final para string Markdown para exibição e download
            final_report_md_string = to_markdown_string(final_report_content)

            # Armazena o relatório final no session_state para que o download button possa acessá-lo
            st.session_state['final_report_md'] = final_report_md_string

            # Exibe o relatório final
            report_container.markdown("## 📝 Relatório Final de Personalidade e Propósito de Vida")
            report_container.markdown(final_report_md_string)

            # Exibe a tabela de sucesso novamente (opcional, já está no relatório final string, mas bom ter como DataFrame)
            if 'sucesso_df' in st.session_state and not st.session_state['sucesso_df'].empty:
                 report_container.markdown("### Pessoas de Sucesso Nascidas na Mesma Data")
                 report_container.dataframe(st.session_state['sucesso_df'])


        except ValueError:
            st.error("Formato de data incorreto. Por favor, use o formato DD/MM/AAAA.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
            # Optional: print traceback for debugging
            # import traceback
            # st.text(traceback.format_exc())

# --- Botão de Download (Aparece APENAS se o relatório foi gerado) ---
# O botão de download deve estar no escopo principal do script para que o Streamlit o renderize
# Ele usará o conteúdo armazenado em st.session_state
if 'final_report_md' in st.session_state and st.session_state['final_report_md']:
     # Adiciona o botão de download abaixo do relatório no container
     report_container.download_button(
         label="💾 Salvar Relatório (Markdown)",
         data=st.session_state['final_report_md'],
         file_name=f"relatorio_personalidade_{data_nascimento_str.replace('/', '-')}.md",
         mime="text/markdown",
         key='download_button' # Chave única para o botão
     )

st.markdown("---")
st.markdown("Desenvolvido com Google AI.")