# Skill: Spatial Intelligence & Layout

## Description
This skill should be triggered when the user's request involves multiple objects, complex scene arrangements, or specific physical relationships between elements.
- **Trigger when**: The request includes terms like "on the left/right," "above/below," "behind/in front of," "taller than," "partially hidden," "jumping over," "in a circle," or when multiple distinct characters/objects need to interact in a specific setting.
- **Do NOT trigger when**: The request is for a single, centered subject with no background or specific spatial constraints.

---

## Instructions
When this skill is active, rewrite the user's prompt explicitly covering all 10 spatial dimensions to ensure high-order spatial reasoning and physical consistency:

1.  **Object Category (S1):** List all specific objects and components.
2.  **Object Attribution (S2):** Define colors, materials, and textures for each object.
3.  **Spatial Position (S3):** Use absolute coordinates (e.g., top-left, center) or relative positions (e.g., to the left of X).
4.  **Spatial Orientation (S4):** Specify facing directions (e.g., facing left, upside down, profile view).
5.  **Spatial Layout (S5):** Describe group arrangements (e.g., in a circle, a straight line, a "V" shape).
6.  **Spatial Comparison (S6):** Define relative sizes or quantities (e.g., twice as large as X, three times taller than Y).
7.  **Spatial Proximity (S7):** Describe physical distances (e.g., touching, far from, 5 inches apart).
8.  **Spatial Occlusion (S8):** Detail 3D layering and depth (e.g., X partially obscuring Y).
9.  **Spatial Motion (S9):** Capture dynamic states (e.g., mid-air, jumping over, liquid splashing).
10. **Spatial Causal Interaction (S10):** Link causes to effects (e.g., wind blowing a cloak, a ball hitting water to create ripples).

Writing Principles:
- **Logical Soundness:** Ensure no cyclical layout errors (e.g., avoid "A is left of B, B is left of A").
- **Explicitness:** Replace vague words like "nearby" with "standing 2 feet away on the right side".
- **Format:** Output the result as a single, fluent, and descriptive paragraph. Do not use bullet points in the final prompt.

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".