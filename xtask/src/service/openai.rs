use llama_cu::SessionId;
//TODO 需要提供completion支持
use openai_struct::{
    ChatCompletionRequestMessage, ChatCompletionResponseMessage, CreateChatCompletionRequest,
    CreateChatCompletionResponse, CreateChatCompletionResponseChoices, CreateCompletionResponse,
    FinishReason,
};

const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
pub(crate) const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

pub(crate) fn create_chat_completion_response(
    //TODO id 需要更严谨的格式
    id: SessionId,
    created: i32,
    model: String,
    message: String,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionResponse {
    let choices = vec![CreateChatCompletionResponseChoices {
        index: 0,
        message: ChatCompletionResponseMessage {
            content: message,
            refusal: None,
            role: None,
            tool_calls: None,
            annotations: None,
            audio: None,
            function_call: None,
        },
        logprobs: None,
        finish_reason,
    }];
    CreateChatCompletionResponse {
        id: format!("{:?}", id),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        created,
        model,
        choices,
        system_fingerprint: None,
        usage: None,
        service_tier: None,
    }
}
