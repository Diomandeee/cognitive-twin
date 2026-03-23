#!/usr/bin/env python3
"""Find Mohamed's stream-of-consciousness messages from Supabase memory_turns."""

import json, re, sys

data = json.load(sys.stdin)
if not isinstance(data, list):
    print(data); sys.exit()

stream_msgs = []
for d in data:
    t = (d.get('content_text') or '').strip()
    if len(t) < 100: continue

    # Exclude system prompts / structured outputs
    if t.startswith('You are a'): continue
    if t.startswith('{'): continue
    if t.startswith('<task-notification'): continue
    if t.startswith('<turn_aborted'): continue
    if t.startswith('CROSS-PANE BRIDGE'): continue
    if 'Return valid JSON' in t: continue
    if '```' in t[:50]: continue
    if t.count('\n') > 20 and len(t) > 2000: continue
    if 'Traceback' in t[:100]: continue
    if 'Uncaught' in t[:100]: continue

    conversational = sum([
        'i think' in t.lower(),
        'we should' in t.lower(),
        'we need' in t.lower(),
        "let's" in t.lower(),
        'perhaps' in t.lower(),
        'what if' in t.lower(),
        'as you know' in t.lower(),
        'in fact' in t.lower(),
        'in any case' in t.lower(),
        'keep in mind' in t.lower(),
        'considering' in t.lower(),
        'the idea' in t.lower(),
        'imagine' in t.lower(),
        'given that' in t.lower(),
        'i want' in t.lower(),
        'i need' in t.lower(),
        'remember' in t.lower(),
        'also' in t.lower(),
        'as well' in t.lower(),
        'the fact that' in t.lower(),
        'to be honest' in t.lower(),
        'i was thinking' in t.lower(),
        'figure out' in t.lower(),
        'how would' in t.lower(),
        'definitely' in t.lower(),
    ])

    sentence_count = len(re.split(r'[.!?]+', t))

    if conversational >= 2 or (sentence_count >= 3 and len(t) > 200):
        stream_msgs.append({
            'text': t,
            'length': len(t),
            'conversational_score': conversational,
            'sentences': sentence_count,
            'depth': d.get('trajectory_depth'),
            'complexity': d.get('trajectory_complexity'),
            'phase': d.get('trajectory_phase_confidence'),
            'date': d.get('created_at', '')[:10]
        })

stream_msgs.sort(key=lambda x: x['conversational_score'] * x['length'], reverse=True)

print(f'Stream of consciousness messages: {len(stream_msgs)}')
print(f'>200 chars: {sum(1 for m in stream_msgs if m["length"] > 200)}')
print(f'>500 chars: {sum(1 for m in stream_msgs if m["length"] > 500)}')
print(f'>1000 chars: {sum(1 for m in stream_msgs if m["length"] > 1000)}')
print()
print('=== TOP 10 ===')
for m in stream_msgs[:10]:
    print(f'{m["length"]:>5} chars | conv={m["conversational_score"]} | {m["date"]}')
    print(f'  {m["text"][:180]}...')
    print()

with open('/Users/mohameddiomande/projects/karl/stream_of_consciousness.json', 'w') as f:
    json.dump(stream_msgs, f)
print(f'Saved {len(stream_msgs)} messages')
