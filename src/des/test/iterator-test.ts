import {LinkedQueue} from "@oresoftware/linked-queue";


const q = new LinkedQueue();


q.enqueue({foo:1});
q.enqueue({foo:2});
q.enqueue({foo:3});
q.enqueue({foo:4});


for(const v of q.iterator()){
  console.log(v);
}

for(const v of q.iterator()){
  console.log(v);
  break;
}