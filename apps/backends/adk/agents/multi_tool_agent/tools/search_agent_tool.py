
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool

from google.adk.tools import google_search



def _render_reference(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """Appends grounding references to the response."""
    del callback_context
    if (
        not llm_response.content or
        not llm_response.content.parts or
        not llm_response.grounding_metadata
    ):
        return llm_response
    references = []
    for chunk in llm_response.grounding_metadata.grounding_chunks or []:
        title, uri, text = '', '', ''
        if chunk.retrieved_context:
            title = chunk.retrieved_context.title
            uri = chunk.retrieved_context.uri
            text = chunk.retrieved_context.text
        elif chunk.web:
            title = chunk.web.title
            uri = chunk.web.uri
        parts = [s for s in (title, text) if s]
        if uri and parts:
            parts[0] = f'[{parts[0]}]({uri})'
        if parts:
            references.append('* ' + ': '.join(parts) + '\n')
    if references:
        reference_text = ''.join(['\n\nReference:\n\n'] + references)
        # llm_response.content.parts.append(types.Part(text=reference_text))

    # grounding_supports
    support_json = [
        support.model_dump_json() for support in llm_response.grounding_metadata.grounding_supports or []
    ]
    if support_json:
        support_text = f"<grounding_supports>{support_json}</grounding_supports>"
        llm_response.content.parts.append(types.Part(text=support_text))

    # grounding_chunks
    chunk_json = [
        chunk.model_dump_json() for chunk in llm_response.grounding_metadata.grounding_chunks or []
    ]
    if chunk_json:
        chunk_text = f"<grounding_chunks>{chunk_json}</grounding_chunks>"
        llm_response.content.parts.append(types.Part(text=chunk_text))

    # search_entry_point
    search_entry_point_json = llm_response.grounding_metadata.search_entry_point.model_dump_json() if llm_response.grounding_metadata.search_entry_point else ""
    if search_entry_point_json:
        search_entry_point_text = f"<search_entry_point>{search_entry_point_json}</search_entry_point>"
        llm_response.content.parts.append(types.Part(text=search_entry_point_text))

    # web_search_queries
    if llm_response.grounding_metadata.web_search_queries:
        web_search_queries_text = f"<web_search_queries>{"|".join(llm_response.grounding_metadata.web_search_queries)}</web_search_queries>"
        llm_response.content.parts.append(types.Part(text=web_search_queries_text))


    if all(part.text is not None for part in llm_response.content.parts):
        all_text = '\n'.join(part.text for part in llm_response.content.parts)
        llm_response.content.parts[0].text = all_text
        del llm_response.content.parts[1:]
    return llm_response

_search_information_gatherer_agent = Agent(
    model="gemini-2.0-flash",
    name="search_information_gatherer_agent",
    description="An agent providing Google-search grounding capability",
    instruction=""",
    Answer the user's question directly using google_search grounding tool. 
    Provide a concise but detailed response. 
    
    Structure your response to include:
    
    * searches your performed
    * sources you found as a list of references. Include links for every reference as a markdown link with the complete URL. You must repeat the URL in the write up - I cannot access the grounding metadata.
    * relevant extracts from those sources including citations
    * commentary on the extracts, reflecting on how they relate to the original question
    * a comprehensive overall summary
    """,
    tools=[google_search],
    after_model_callback=_render_reference,
)

_END_OF_EDIT_MARK = '---END-OF-EDIT---'


def _remove_end_of_edit_mark(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    del callback_context  # unused
    if not llm_response.content or not llm_response.content.parts:
        return llm_response
    for idx, part in enumerate(llm_response.content.parts):
        if _END_OF_EDIT_MARK in part.text:
            del llm_response.content.parts[idx + 1 :]
            part.text = part.text.split(_END_OF_EDIT_MARK, 1)[0]
    return llm_response


_google_search_report_writer = Agent(
    model="gemini-2.0-flash",
    name="google_search_report_writer",
    description="An agent for writing reports on the search results",
    instruction="""
    Refine the answer to the users question.
    Provide a concise but detailed response.

    You will receive a response from another agent - you need to refine this response. Structure your response to include:

    * searches your performed
    * sources you found as a list of references. Include links for every reference as a markdown link with the complete URL. You must repeat the URL in the write up - I cannot access the grounding metadata.
    * relevant extracts from those sources including citations
    * commentary on the extracts, reflecting on how they relate to the original question
    * a comprehensive overall summary
    
    It is crucial that in your re-write you incorporate the references as inline markdown links to support the answer. The resulting answer should be a well formatted report suitable for displaying to the user via a markdown formatter.
    """,
    tools=[],
    after_model_callback=_remove_end_of_edit_mark,
)

_search_agent = SequentialAgent(
    name="google_search_sequence",
    description="An agent for writing reports on the search results. The  results will be displayed in a user friendly markdown format. Links will be available for citations inline in markdown format.",
    sub_agents=[
        _search_information_gatherer_agent,
        _google_search_report_writer
    ],
)

google_search_agent_tool = AgentTool(agent=_search_agent)
