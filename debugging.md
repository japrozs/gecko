## while writing `Value`

- make sure in `backward` fn, you are adding to the gradient and not rewriting it (`self.grad += ` instead of `self.grad = `) and you're attaching the backward function to the output
