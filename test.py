# def test_stack():
#     stack_list = []

#     def reversive(typ, count):
#         print(f"typ: {typ}, count: {count}, stack_list: {stack_list}")

#         if count > 5:
#             return stack_list
#         stack_list.append(count * 2)
#         reversive("A", count + 1)
#         stack_list.pop()
#         stack_list.append(count * 3)
#         reversive("B", count + 1)
#         stack_list.pop()
    
#     reversive( "A", 0) 
#     return stack_list

def generate_parenthesis(n:int) -> str:

    result = []
    left=right=0
    q = [(left, right, '')]
    while q:
        left, right, s = q.pop(0)
        if left==n and right==n:
            result.append(s)
        
        if left<n:
            q.append((left+1, right, s+'('))
        if right<left:
            q.append((left, right+1, s+')'))
    return result


if __name__ == "__main__":
    output = generate_parenthesis(3)
    print(f"from main funtion: {output}")