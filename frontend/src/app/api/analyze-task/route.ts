import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';

export async function POST(req: NextRequest) {
  const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  const { task_description } = await req.json();

  const prompt = `You are a task analysis assistant. Analyze the following task and provide three metrics and a step breakdown:
1. "complexity": a score from 1 to 5 evaluating how complex the task is.
2. "steps": an integer estimating how many sub-steps the task will take.
3. "priority": an integer from 1 to 5 evaluating how high priority the task is.
4. "steps_list": an array of strings, where each string is a concise step to complete the task.

Task: ${task_description}

Return ONLY a valid JSON object matching this schema exactly: {"complexity": 0, "steps": 0, "priority": 0, "steps_list": ["step 1", "step 2"]}.`;

  try {
    const response = await groq.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: 'llama-3.1-8b-instant',
      temperature: 0.1,
      response_format: { type: 'json_object' },
    });

    const parsed = JSON.parse(response.choices[0].message.content ?? '{}');
    return NextResponse.json({
      complexity:  parsed.complexity  ?? 3,
      steps:       parsed.steps       ?? 10,
      priority:    parsed.priority    ?? 3,
      steps_list:  parsed.steps_list  ?? [],
    });
  } catch (e) {
    console.error('Groq error:', e);
    return NextResponse.json({ complexity: 3, steps: 10, priority: 3, steps_list: [] });
  }
}
